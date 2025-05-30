import jax
import jax.numpy as jnp
from rodeo.kalmantv import standard
import equinox as eqx
from svi.utils import theta_to_chol


# RNN for forward pass
class RNN(eqx.Module):
    hidden_size: int
    layers: list
    linear: eqx.Module
     
    def __init__(self, key, n_state, n_meas):
        key, *subkey = jax.random.split(key, num=4)
        self.hidden_size = n_state*(3 + 2*n_state)* 2 + 4 * n_state * 3
        self.hidden_size = self.hidden_size //2
        self.layers = [
            eqx.nn.GRUCell(n_meas, self.hidden_size, key=subkey[0]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[1]),
            # eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[2]),
            # eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[3]),
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[2])
    

    # GRU(y_t,h_t) -> h_{t+1}
    def __call__(self, y_meas):
        hidden = jnp.zeros((len(self.layers), self.hidden_size,))
        data_seq = y_meas
        for i in range(len(hidden)):
            def f(carry, inp):
                return self.layers[i](inp, carry), self.layers[i](inp, carry)
            final, data_seq = jax.lax.scan(f, hidden[i], data_seq)
        out = jax.vmap(self.linear)(data_seq)
        return out

# NN for backward pass
class NN(eqx.Module):
    hidden_size: int
    out_size: int
    layers: list
    linear: eqx.Module
     
    def __init__(self, key, n_state):
        key, *subkey = jax.random.split(key, num=5)
        self.out_size = n_state + n_state*(n_state + 1)//2
        n_inp = n_state*n_state + 2*n_state + n_state*(n_state + 1)//2
        self.hidden_size = 50
        self.layers = [
            eqx.nn.Linear(n_inp, n_inp, key=subkey[0]),
            jax.nn.relu,
            eqx.nn.Linear(n_inp, n_inp, key=subkey[1]),
            jax.nn.relu,
            eqx.nn.Linear(n_inp, n_inp, key=subkey[2]),
            jax.nn.relu
        ]
        self.linear = eqx.nn.Linear(n_inp, self.out_size, key=subkey[3])

    def __call__(self, input):
        
        for layer in self.layers:
            input = layer(input)
        
        out = self.linear(input)
        return out

# NN for random effects
class NN_rand(eqx.Module):
    hidden_size: int
    out_size: int
    layers: list
    linear: eqx.Module
     
    def __init__(self, key, n_theta, n_obs, n_effect):
        key, *subkey = jax.random.split(key, num=5)
        self.out_size = n_effect * n_theta + n_effect + n_effect * (n_effect + 1) // 2
        n_inp = n_obs
        self.hidden_size = 20
        self.layers = [
            eqx.nn.Linear(n_inp, 4*n_inp, key=subkey[0]),
            jax.nn.relu,
            eqx.nn.Linear(4*n_inp, 8*n_inp, key=subkey[1]),
            jax.nn.relu,
            eqx.nn.Linear(8*n_inp, 4*n_inp, key=subkey[2]),
            jax.nn.relu,

        ]
        self.linear = eqx.nn.Linear(4*n_inp, self.out_size, key=subkey[3])

    def __call__(self, input):
        
        for layer in self.layers:
            input = layer(input)
        
        out = self.linear(input)
        return out

class SmoothModel:
    r"""
    The variational distribution is given by
    \begin{equation}
    \begin{aligned}
        \theta \mid y_{0:N} &\sim N(\mu(y_{0:N}), \Sigma(y_{0:N})) \\
        \eta \mid y_{0:N}, \theta &\sim N(A(y_{0:N}) \theta + b(y_{0:N}), V(y_{0:N})) \\
        x_{0:N} \mid y_{0:N}, \eta  &\sim N\big(\mu_{x} (y_{0:N}, \eta), \Sigma_{x}(y_{0:N}, \eta)\big).
    \end{aligned}
    \end{equation}
    where $x$ are the latents, $y$ are the observations, $\theta$ are the SDE parameters, and $\eta$ are the random effects.
    The random-effects are modelled via a neural network given by
    $$
    A, b, V = \textnormal{NN}_{\eta}(y_{0:N}).
    $$
    The latents are modelled by a RNN give by
    $$
    \mu_{x}, \Sigma_{x} = \textnormal{RNN}(y_{0:N}, \eta)
    $$
    by fitting the Kalman filtering parameters. On the backward pass we use the conditional Gaussian model given by
    \begin{equation}
    \begin{aligned}
        x_n \mid x_{n+1} \sim N(\mu^\prime_n, \Sigma^\prime_n)
    \end{aligned}
    \end{equation}
    where we fit a neural network to find
    $$
    \mu^\prime_n, \Sigma^\prime_n, = \textnormal{NN}(x_{n+1}, \mu_{x}, \Sigma_{x}).
    $$

    Args:
        n_state (int): Dimension of the latent at each time point.
        random_ind (jnp.array): The indices of the parameters used to generate the random effects.
        fixed_ind (jnp.array): The indices of the fixed-effect parameter.
        obs_times (jnp.array): The observation time points.
        sde_times (jnp.array): The discretization time points to sample the latents.
    """

    def __init__(self, n_state, random_ind, fixed_ind, obs_times, sde_times):
        self._n_state = n_state
        self._random_ind = random_ind
        self._n_random = len(random_ind)//2
        self._fixed_ind = fixed_ind
        self._n_phi = self._n_random + len(self._fixed_ind)
        self._obs_times = obs_times
        self._sde_times = sde_times
        self._dt = sde_times[1] - sde_times[0]
        self._n_sde = len(sde_times)

    def _rnn_input(self, y_meas):
        y_meas = jnp.atleast_2d(y_meas)
        # theta_rep = jnp.repeat(theta[None], self._n_sde, axis=0)
        obs_ind = jnp.searchsorted(self._obs_times, self._sde_times[:-1], side='right')
        time_prev = self._obs_times[obs_ind-1]
        time_diff = self._sde_times[:-1] - time_prev
        time_diff = jnp.append(time_diff, 0)
        y_meas_prev = y_meas[obs_ind-1]
        y_meas_next = y_meas[obs_ind]
        y_meas_comb = jnp.hstack([y_meas_prev, y_meas_next])
        y_meas_last = jnp.append(y_meas[-1], y_meas[-1])
        y_meas_comb = jnp.vstack([y_meas_comb, y_meas_last])
        input = jnp.hstack([y_meas_comb, time_diff[:, None]])
        return input

    def _par_parse(self, params, y_meas):
        gru_model = params["gru"]
        full_par = gru_model(y_meas)
        n_phi = self._n_phi
        # split parameters into mean_state_filt, mean_state, wgt_state, var_state_filt, var_state
        par_indices = [self._n_state, 2*self._n_state, self._n_state*(2+self._n_state+n_phi), self._n_state*(3*self._n_state+5+2*n_phi)//2, self._n_state*(2*self._n_state+3+n_phi)]
        par_indices = [x + n_phi * self._n_state for x in par_indices]
        mean_state_filt = full_par[:, :par_indices[0]]
        mean_state_filt1 = mean_state_filt[:, :self._n_state]
        mean_state_filt2 = mean_state_filt[:, self._n_state:].reshape((self._n_sde, self._n_state, n_phi)) * 0.01
        mean_state = self._dt * full_par[:, par_indices[0]:par_indices[1]]  
        wgt_state = full_par[:, par_indices[1]:par_indices[2]].reshape(self._n_sde, self._n_state, self._n_state + n_phi)
        wgt_state1 = wgt_state[:, :, :self._n_state]
        wgt_state2 = wgt_state[:, :, self._n_state:]
        upper_ind = jnp.triu_indices(self._n_state)
        diag_ind = jnp.diag_indices(self._n_state)
        chol_state_filt = jnp.zeros((self._n_state, self._n_state, self._n_sde))
        chol_state_filt = chol_state_filt.at[upper_ind].set(full_par[:, par_indices[2]:par_indices[3]].T)
        chol_state_filt = chol_state_filt.at[diag_ind].set(jax.nn.softplus(chol_state_filt[diag_ind])).T
        chol_state = jnp.zeros((self._n_state, self._n_state, self._n_sde))
        chol_state = chol_state.at[upper_ind].set(full_par[:, par_indices[3]:par_indices[4]].T)
        chol_state = chol_state.at[diag_ind].set(jax.nn.softplus(chol_state[diag_ind])).T
        # convert cholesky to variance
        def chol_to_var(chol_mat):
            var_mat = chol_mat.dot(chol_mat.T)
            return var_mat
        var_state_filt = jax.vmap(chol_to_var)(chol_state_filt)
        var_state = self._dt * jax.vmap(chol_to_var)(chol_state)
        
        # mean_state = mean_state + wgt_state2.dot(random_fixed) # do not save this because phi changes during inference
        # compute predicted values
        mean_state_pred, var_state_pred = jax.vmap(standard.predict)(
            mean_state_past=mean_state_filt1,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            wgt_state=wgt_state1,
            var_state=var_state
        )
        return mean_state_pred, var_state_pred, mean_state_filt1, mean_state_filt2, var_state_filt, wgt_state1, wgt_state2

    def _sim_random(self, key, model, theta, y_meas):
        n_sim, n_obs = y_meas.shape[0:2]        
        random_normals = jax.random.normal(key, shape=(n_sim, self._n_random))
        n_theta = len(theta) 
        # model = params["nn_random"]
        def vmap_fun(random_normal, y_n):
            model_output = model(y_n.flatten())
            # y_theta = jnp.append(y_n, theta)
            # model_output = model(y_theta)
            wgt_ind = self._n_random * n_theta
            wgt_theta = model_output[:wgt_ind].reshape(self._n_random, n_theta)*0.01 + jnp.eye(self._n_random, n_theta)
            mu_theta = model_output[wgt_ind:wgt_ind+self._n_random]
            lower_theta = model_output[wgt_ind+self._n_random:]
            chol_theta = theta_to_chol(lower_theta, self._n_random)
            random_mu = wgt_theta.dot(theta) + mu_theta
            random_effect = random_mu + chol_theta.dot(random_normal)
            nlp = -jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(random_effect, random_mu, chol_theta.dot(chol_theta.T)))
            return random_effect, nlp
        
        random_effect, nlp = jax.vmap(vmap_fun)(random_normals, y_meas)
        return random_effect, jnp.sum(nlp)

    def _back_sim(self, key, mean_state_pred, var_state_pred,
                  mean_state_filt, var_state_filt, wgt_state):
        
        # simulate using backward Markov Chain
        def scan_fun(carry, fwd_kwargs):
            mean_state_filt = fwd_kwargs['mean_state_filt']
            var_state_filt = fwd_kwargs['var_state_filt']
            mean_state_pred = fwd_kwargs['mean_state_pred']
            var_state_pred = fwd_kwargs['var_state_pred']
            wgt_state = fwd_kwargs['wgt_state']
            random_normal = fwd_kwargs['random_normal']
            x_state_next = carry['x_state_next']
            x_neglogpdf = carry["x_neglogpdf"]
            # get Markov params
            mean_state_sim, var_state_sim = standard.smooth_sim(
                x_state_next=x_state_next,
                wgt_state=wgt_state,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred
            )
            chol_factor = jnp.linalg.cholesky(var_state_sim)
            x_state_curr = mean_state_sim + chol_factor.dot(random_normal)
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(x_state_curr, mean_state_sim, var_state_sim)
            carry = {
                'x_state_next': x_state_curr,
                'x_neglogpdf': x_neglogpdf
            }
            return carry, carry
        
        # time N
        # mean_state_N = mean_state_filt[self._n_sde-1] + self._x_init
        mean_state_N = mean_state_filt[self._n_sde-1] 
        var_state_N = var_state_filt[self._n_sde-1]
        random_normals = jax.random.normal(key, shape=(self._n_sde, self._n_state))
        chol_factor = jnp.linalg.cholesky(var_state_N)
        x_N = mean_state_N + chol_factor.dot(random_normals[self._n_sde-1])
        x_neglogpdf = -jax.scipy.stats.multivariate_normal.logpdf(x_N, mean_state_N, var_state_N)
        scan_init = {
            'x_state_next': x_N,
            'x_neglogpdf': x_neglogpdf
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_filt': mean_state_filt[:self._n_sde-1],
            'var_state_filt': var_state_filt[:self._n_sde-1],
            'mean_state_pred': mean_state_pred[:self._n_sde-1],
            'var_state_pred': var_state_pred[:self._n_sde-1],
            'wgt_state': wgt_state[:self._n_sde-1],
            'random_normal': random_normals[:self._n_sde-1]
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)
        x_state_smooth = jnp.concatenate(
            [stack_out['x_state_next'], x_N[None]]
        )
        x_neglogpdf = last_out["x_neglogpdf"]
        return x_state_smooth, x_neglogpdf

    def _back_mode(self, mean_state_pred, var_state_pred,
                  mean_state_filt, var_state_filt, wgt_state):
        
        # simulate using backward Markov Chain
        def scan_fun(carry, fwd_kwargs):
            mean_state_filt = fwd_kwargs['mean_state_filt']
            var_state_filt = fwd_kwargs['var_state_filt']
            mean_state_pred = fwd_kwargs['mean_state_pred']
            var_state_pred = fwd_kwargs['var_state_pred']
            wgt_state = fwd_kwargs['wgt_state']
            x_state_next = carry['x_state_next']
            x_neglogpdf = carry["x_neglogpdf"]
            # get Markov params
            mean_state_mv, var_state_mv = standard.smooth_sim(
                x_state_next=x_state_next,
                wgt_state=wgt_state,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred
            )
            x_state_curr = mean_state_mv
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(x_state_curr, mean_state_mv, var_state_mv)
            carry = {
                'x_state_next': x_state_curr,
                'x_neglogpdf': x_neglogpdf
            }
            return carry, carry
        
        # time N
        # mean_state_N = mean_state_filt[self._n_sde-1] + self._x_init
        mean_state_N = mean_state_filt[self._n_sde-1] 
        var_state_N = var_state_filt[self._n_sde-1]
        x_N = mean_state_N 
        x_neglogpdf = -jax.scipy.stats.multivariate_normal.logpdf(x_N, mean_state_N, var_state_N)
        scan_init = {
            'x_state_next': x_N,
            'x_neglogpdf': x_neglogpdf
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_filt': mean_state_filt[:self._n_sde-1],
            'var_state_filt': var_state_filt[:self._n_sde-1],
            'mean_state_pred': mean_state_pred[:self._n_sde-1],
            'var_state_pred': var_state_pred[:self._n_sde-1],
            'wgt_state': wgt_state[:self._n_sde-1],
        }

        last_out, stack_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)
        x_state_smooth = jnp.concatenate(
            [stack_out['x_state_next'], x_N[None]]
        )
        x_neglogpdf = last_out["x_neglogpdf"]
        return x_state_smooth, x_neglogpdf


    def _sim_one(self, key, params, random_fixed, y_meas):
        obs_input = self._rnn_input(y_meas)
        mean_state_pred, var_state_pred, \
            mean_state_filt1, mean_state_filt2, \
                var_state_filt, wgt_state1, wgt_state2 = self._par_parse(params, obs_input)
        state_filt_random = jax.vmap(jnp.dot, in_axes=[0, None])(mean_state_filt2, random_fixed)
        mean_state_pred = mean_state_pred + jax.vmap(jnp.dot, in_axes=[0, None])(wgt_state2, random_fixed) + \
            jax.vmap(jnp.dot)(wgt_state1, state_filt_random)
        mean_state_filt = mean_state_filt1 + state_filt_random
        
        return self._back_sim(key, mean_state_pred, var_state_pred, 
                              mean_state_filt,var_state_filt, wgt_state1)
        
    def _simulate_theta(self, key, params):
        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)
        
        # theta_std = jax.nn.softplus(params["theta_std"])
        theta_normal = jax.random.normal(key, shape=(n_theta,))
        theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        theta = theta_mu + theta_chol.dot(theta_normal)
        # theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.diag(theta_chol)))
        theta_entpy = -jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(theta, theta_mu, theta_chol.dot(theta_chol.T)))
        return theta, theta_entpy

    def _sim_random_non(self, key, theta, params):
        n_sim = len(self._sde_times)
        random_wgt = params["random_wgt"]
        random_off = params["random_off"]
        random_chol = theta_to_chol(params["random_chol"], self._n_random)
        random_var = random_chol.dot(random_chol.T)
        random_normals = jax.random.normal(key, shape=(n_sim, self._n_random))
        # model = params["nn_random"]
        def vmap_fun(random_normal):
            random_mu = random_wgt.dot(theta) + random_off
            random_effect = random_mu + random_chol.dot(random_normal)
            nlp = -jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(random_effect, random_mu, random_var))
            return random_effect, nlp
        
        random_effect, nlp = jax.vmap(vmap_fun)(random_normals)
        return random_effect, jnp.sum(nlp)

    def _save_pars(self, params, y_meas):

        def vmap_fun(y_n):
            obs_input = self._rnn_input(y_n)
            return self._par_parse(params, obs_input)
            
        self.mean_state_pred, self.var_state_pred, \
            self.mean_state_filt1, self.mean_state_filt2, self.var_state_filt, \
            self.wgt_state1, self.wgt_state2 = jax.vmap(vmap_fun)(y_meas)
    
    def _sim_x(self, key, mean_state_pred, var_state_pred, mean_state_filt1, 
               mean_state_filt2, var_state_filt, wgt_state1, wgt_state2, phi, x_init):
        
        state_filt_random = jax.vmap(jnp.dot, in_axes=[0, None])(mean_state_filt2, phi)
        mean_state_pred = mean_state_pred + jax.vmap(jnp.dot, in_axes=[0, None])(wgt_state2, phi) + \
            jax.vmap(jnp.dot)(wgt_state1, state_filt_random)
        # state_filt_random = mean_state_filt2.dot(phi)
        # mean_state_pred = mean_state_pred + wgt_state2.dot(phi) + wgt_state1.dot(state_filt_random)
        mean_state_filt = mean_state_filt1 + state_filt_random

        x_state_smooth, x_neglogpdf = self._back_sim(key, mean_state_pred, var_state_pred, 
                                                     mean_state_filt, var_state_filt, wgt_state1)
        return x_state_smooth + x_init, x_neglogpdf

    def _sim_trained(self, key, random_fixed, x_init):
        
        return jax.vmap(self._sim_x)(key, self.mean_state_pred, self.var_state_pred, 
                                     self.mean_state_filt1, self.mean_state_filt2, self.var_state_filt,
                                     self.wgt_state1, self.wgt_state2, random_fixed, x_init)


    def simulate(self, key, params, y_meas, x_init):
        r"""
        Simulate $\theta$, $\eta$ and $x$ using the variational distribution. Also compute $\log q(\theta) + \log q(\eta \mid \theta)  + \log q(x \mid \eta)$.

        Args:
            key (random.key): PRNG key.
            params (dict): Dictionary to hold the neural network parameters and other parameters necessary to simulate $\theta$ and $x$.
            y_meas (jnp.array): Observations.
            x_init (jnp.array): Initial guess of the latent.

        Returns:
            x (jnp.array): Latents at the SDE time points.
            theta (jnp.array): SDE parameters.
            eta (jnp.array): Random effects.
            logpdf (float):  $\log q(\theta) + \log q(\eta \mid \theta)  + \log q(x \mid \eta)$.
        """
        key, *subkeys = jax.random.split(key, num=4)
        # simulate theta
        theta, theta_entpy = self._simulate_theta(subkeys[0], params)
        idx = jnp.array([1,3,4])
        theta_exp = theta.at[idx].set(jnp.exp(theta[idx])) # convert to original scale
        
        # simulate random effect
        # nn version
        nn_random_model = params["nn_random"]
        random_effect, random_neglogpdf = self._sim_random(subkeys[1], nn_random_model, theta_exp, y_meas)

        # non-nn version
        # random_effect, random_neglogpdf = self._sim_random_non(subkeys[1], theta_exp, params)

        fixed_effect = theta_exp[self._fixed_ind]
        def vmap_fun(random_effect_n, y_n):
            random_fixed = jnp.append(random_effect_n, fixed_effect)
            x_state, x_neglogpdf = self._sim_one(subkeys[2], params, random_fixed, y_n)
            return x_state, x_neglogpdf
        x_state, x_neglogpdf = jax.vmap(vmap_fun)(random_effect, y_meas)
        x_neglogpdf = jnp.sum(x_neglogpdf)
        # calculate -E[log q(x, theta|phi_full)]
        # use entropy for - E[log q(theta)]
        # use negative logpdf for - E[log q(eta|theta)]
        # use negative logpdf for - E[log q(x|eta)]
        theta_x_neglogpdf = x_neglogpdf + theta_entpy + random_neglogpdf
        return (x_state + x_init, theta, random_effect), theta_x_neglogpdf
