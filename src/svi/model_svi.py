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
        self.hidden_size = n_state*(3 + 2*n_state)*2
        self.layers = [
            eqx.nn.GRUCell(n_meas, self.hidden_size, key=subkey[0]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[1]),
            # eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[2]),
            # eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[3]),
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, self.hidden_size//2, key=subkey[2])
    

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
        key, *subkey = jax.random.split(key, num=6)
        self.out_size = n_state + n_state*(n_state + 1)//2
        n_inp = n_state*n_state + 2*n_state + n_state*(n_state + 1)//2
        self.hidden_size = 50
        self.layers = [
            eqx.nn.Linear(n_inp, self.hidden_size, key=subkey[0]),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[1]),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[2]),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[3]),
            jax.nn.relu
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, self.out_size, key=subkey[4])

    

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
        x_{0:N} \mid y_{0:N}, \theta  &\sim N\big(\mu_{x} (y_{0:N}, \theta), \Sigma_{x}(y_{0:N}, \theta)\big).
    \end{aligned}
    \end{equation}
    where $x$ are the latents, $y$ are the observations, and $\theta$ are the SDE parameters.
    The underlying RNN provides
    $$
    \mu_{x}, \Sigma_{x} = \textnormal{RNN}(y_{0:N}, \theta)
    $$
    by fitting the Kalman filtering parameters. On the backward pass we use the conditional Gaussian model given by
    \begin{equation}
    \begin{aligned}
        x_n \mid x_{n+1} &\sim N(\mu^\prime_n, \Sigma^\prime_n)
    \end{aligned}
    \end{equation}
    where we fit a neural network to find
    $$
    \mu^\prime_n, \Sigma^\prime_n, = \textnormal{NN}(x_{n+1}, \mu_{x}, \Sigma_{x}).
    $$

    Args:
        n_state (int): Dimension of the latent at each time point.
        obs_times (jnp.array): The observation time points.
        sde_times (jnp.array): The discretization time points to sample the latents.
    """

    def __init__(self, n_state, obs_times, sde_times):
        self._n_state = n_state
        self._obs_times = obs_times
        self._sde_times = sde_times
        self._dt = sde_times[1] - sde_times[0]
        self._n_sde = len(sde_times)

    def _rnn_input(self, theta, y_meas):
        theta_rep = jnp.repeat(theta[None], self._n_sde, axis=0)
        obs_ind = jnp.searchsorted(self._obs_times, self._sde_times[:-1], side='right')
        time_prev = self._obs_times[obs_ind-1]
        time_diff = self._sde_times[:-1] - time_prev
        time_diff = jnp.append(time_diff, 0) * 10
        y_meas_prev = y_meas[obs_ind-1]
        y_meas_next = y_meas[obs_ind]
        y_meas_comb = jnp.hstack([y_meas_prev, y_meas_next])
        y_meas_last = jnp.append(y_meas[-1], y_meas[-1])
        y_meas_comb = jnp.vstack([y_meas_comb, y_meas_last])
        input = jnp.hstack([y_meas_comb, time_diff[:, None], theta_rep])
        return input

    def _par_parse(self, params, obs_theta):
        gru_model = params["gru"]
        full_par = gru_model(obs_theta)
        # split parameters into mean_state_filt, mean_state, wgt_state, var_state_filt, var_state
        par_indices = [self._n_state, 2*self._n_state, self._n_state*(2+self._n_state), self._n_state*(3*self._n_state+5)//2, self._n_state*(2*self._n_state+3)]
        mean_state_filt = full_par[:, :par_indices[0]]
        mean_state = self._dt * full_par[:, par_indices[0]:par_indices[1]] 
        wgt_state = full_par[:, par_indices[1]:par_indices[2]].reshape(self._n_sde, self._n_state, self._n_state) 
        upper_ind = jnp.triu_indices(self._n_state)
        diag_ind = jnp.diag_indices(self._n_state)
        chol_state_filt = jnp.zeros((self._n_state, self._n_state, self._n_sde)) 
        chol_state_filt = chol_state_filt.at[upper_ind].set(full_par[:, par_indices[2]:par_indices[3]].T)*0.1
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

        # compute predicted values
        mean_state_pred, var_state_pred = jax.vmap(standard.predict)(
            mean_state_past=mean_state_filt,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            wgt_state=wgt_state,
            var_state=var_state
        )
        return mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, wgt_state

    def simulate_theta(self, key, params):
        r"""
        Simulate $\theta$ using the variation distribution. Also compute $\log q(\theta)$.
        """
        
        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)
        # theta_std = jax.nn.softplus(params["theta_std"])
        random_normal = jax.random.normal(key, shape=(n_theta,))
        # theta = theta_mu + theta_std*random_normal
        theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        theta = theta_mu + theta_chol.dot(random_normal)
        theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.diag(theta_chol)))
        return theta, theta_entpy

    def simulate(self, key, params, y_meas, x_init, theta):
        r"""
        Simulate $x$ using the variational distribution. Also compute $\log q(x \mid \theta)$.

        Args:
            key (random.key): PRNG key.
            params (dict): Dictionary to hold the neural network parameters and other parameters necessary to simulate $x$.
            y_meas (jnp.array): Observations.
            x_init (jnp.array): Initial guess for the latents.
            theta (jnp.array): SDE parameters.

        Returns:
            x (jnp.array): Latents at the SDE time points.
            logpdf (float): $\log q(x \mid \theta)$.
        """
        obs_theta = self._rnn_input(theta, y_meas)
        mean_state_filt, var_state_filt, \
            mean_state_pred, var_state_pred, \
                wgt_state = self._par_parse(params, obs_theta)

        # NN model for the backward pass (conditional MVN)
        nn_model = params["nn"]
        lower_ind = jnp.tril_indices(self._n_state)
        diag_ind = jnp.diag_indices(self._n_state)

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
            wgt_state_back, mean_state_back, var_state_back = standard.smooth_cond(
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                wgt_state=wgt_state
            )
            chol_back = jnp.linalg.cholesky(var_state_back)
            nn_input = jnp.concatenate([wgt_state_back.flatten(), mean_state_back, chol_back[lower_ind], x_state_next])
            nn_output = nn_model(nn_input)
            mean_state_curr = nn_output[:self._n_state]
            chol_curr = jnp.zeros((self._n_state, self._n_state))
            chol_curr = chol_curr.at[lower_ind].set(nn_output[self._n_state:])
            chol_curr = chol_curr.at[diag_ind].set(jax.nn.softplus(chol_curr[diag_ind]))
            var_state_curr = chol_curr.dot(chol_curr.T)
            x_state_curr = mean_state_curr + chol_curr.dot(random_normal) 
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(x_state_curr, mean_state_curr, var_state_curr)
            carry = {
                'x_state_next': x_state_curr,
                'x_neglogpdf': x_neglogpdf
            }
            return carry, carry
        
        # time N
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
        ) + x_init
        # calculate -E[log q(x, theta|phi_full)]
        # use entropy for - E[log q(theta)]
        # use negative logpdf for - E[log q(x|theta)]
        x_neglogpdf = last_out["x_neglogpdf"]
        # theta_entpy = -jax.scipy.stats.multivariate_normal.logpdf(theta, theta_mu, theta_chol.dot(theta_chol.T))
        # theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(theta_std))
        return x_state_smooth, x_neglogpdf
