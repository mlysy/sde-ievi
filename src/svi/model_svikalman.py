import jax
import jax.numpy as jnp
from rodeo.kalmantv import standard
import equinox as eqx
from svi.utils import theta_to_chol, chol_to_var
from itertools import accumulate

# RNN for forward pass
class RNN(eqx.Module):
    n_state: int
    n_res: int
    hidden_size: int
    rnn_layers: list
    lin_layers: list
    final: list
    linear: eqx.Module
    lnorm: eqx.Module
     
    def __init__(self, key, n_state, n_data, n_theta, n_res):
        subkey = jax.random.split(key, num=8)
        self.n_state = n_state
        self.n_res = n_res
        self.hidden_size = 3*n_state*(n_state + 1)*2 
        out_size = self.hidden_size//2 * n_res
        # self.hidden_size = self.hidden_size //2
        self.rnn_layers = [
            eqx.nn.GRUCell(n_data, self.hidden_size, key=subkey[0]),
            eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[1]),
            # eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[2]),
            # eqx.nn.GRUCell(self.hidden_size, self.hidden_size, key=subkey[3]),
        ]
        self.lin_layers = [
            # eqx.nn.LayerNorm(n_theta),
            eqx.nn.Linear(n_theta, self.hidden_size, key=subkey[2]),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, out_size, key=subkey[3]),
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, out_size, key=subkey[5])
        self.lnorm = eqx.nn.LayerNorm(n_data)
        self.final = [
            eqx.nn.LayerNorm(2 * out_size),
            eqx.nn.Linear(2 * out_size, self.hidden_size, key=subkey[6]),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, out_size, key=subkey[7])
        ]

    def __call__(self, data_seq, theta, mean_state_init, var_state_init, wgt_meas, var_meas, y_meas):
        hidden = jnp.zeros((len(self.rnn_layers), self.hidden_size,))
        par_ind = [self.n_state * self.n_res, self.n_state * self.n_state * self.n_res, self.n_state * (self.n_state + 1) * self.n_res // 2]
        par_indices = list(accumulate(par_ind))
        upper_ind = jnp.triu_indices(self.n_state)
        diag_ind = jnp.diag_indices(self.n_state)
        
        for layer in self.lin_layers:
            theta = layer(theta)

        # GRU(y_t,h_t) -> h_{t+1}
        def f(carry, fargs):
            hidden = carry["hidden"]
            mean_state_filt, var_state_filt = carry["state_filt"]
            hidden_next = jnp.zeros_like(hidden)
            data_i = fargs["data_seq"]
            # x_meas = data_i[:len(data_i)//2]
            x_meas = fargs["x_meas"]
            # x_init = fargs["x_init"]
            inp = jnp.concatenate([data_i, mean_state_filt, var_state_filt[upper_ind]])
            # inp = self.lnorm(inp)
            for i in range(len(hidden)):
                inp = self.rnn_layers[i](inp, hidden[i])
                hidden_next = hidden_next.at[i].set(inp)
            data_out = self.linear(inp)
            out = jnp.append(data_out, theta)
            for layer in self.final:
                out = layer(out)
            # out = data_out
            # unpack Q, c, R triples (n_res)
            mean_state = out[:par_indices[0]].reshape(self.n_res, self.n_state)
            wgt_state = out[par_indices[0]:par_indices[1]].reshape(self.n_res, self.n_state, self.n_state) 
            chol_state = jnp.zeros((self.n_state, self.n_state, self.n_res))
            chol_pars = out[par_indices[1]:par_indices[2]].reshape(self.n_res, -1).T
            chol_state = chol_state.at[upper_ind].set(chol_pars) 
            chol_state = chol_state.at[diag_ind].set(jax.nn.softplus(chol_state[diag_ind])).T 
            var_state = jax.vmap(lambda x: x.dot(x.T))(chol_state)
            
            # kalman filtering 
            # need to scan over n_res axis where there is no obs
            def kalman_scan(carry, kwargs):
                # kalman predict
                mean_state_filt, var_state_filt = carry["state_filt"]
                mean_state = kwargs["mean_state"]
                wgt_state = kwargs["wgt_state"]
                var_state = kwargs["var_state"]
                mean_state_pred, var_state_pred = standard.predict(
                    mean_state_past=mean_state_filt,
                    var_state_past=var_state_filt,
                    mean_state=mean_state,
                    wgt_state=wgt_state,
                    var_state=var_state
                )
                # output
                carry = {
                    "state_filt": (mean_state_pred, var_state_pred)
                }
                stack = {
                    "state_filt": (mean_state_pred, var_state_pred),
                    "state_pred": (mean_state_pred, var_state_pred)
                }
                return carry, stack
            
            kwargs = {
                "mean_state": mean_state[:self.n_res-1],
                "wgt_state": wgt_state[:self.n_res-1],
                "var_state": var_state[:self.n_res-1]
            }
            kalman_init = {
                "state_filt": (mean_state_filt, var_state_filt)
            }
            final_out, scan_out = jax.lax.scan(kalman_scan, kalman_init, kwargs)
            # final kalman filter for obs
            # kalman predict
            mean_state_pred, var_state_pred = standard.predict(
                    mean_state_past=final_out["state_filt"][0],
                    var_state_past=final_out["state_filt"][1],
                    mean_state=mean_state[-1],
                    wgt_state=wgt_state[-1],
                    var_state=var_state[-1]
                )
            # kalman update
            mean_state_next, var_state_next = standard.update(
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                x_meas=x_meas,
                mean_meas=jnp.zeros_like(x_meas),
                wgt_meas=wgt_meas,
                var_meas=var_meas
            )
            # mean_state_next = mean_state_pred
            # var_state_next = var_state_pred
            scan_out["state_filt"] = (
                jnp.concatenate([scan_out["state_filt"][0], mean_state_next[None]]),
                jnp.concatenate([scan_out["state_filt"][1], var_state_next[None]])
            )
            scan_out["state_pred"] = (
                jnp.concatenate([scan_out["state_pred"][0], mean_state_pred[None]]),
                jnp.concatenate([scan_out["state_pred"][1], var_state_pred[None]])
            )
            scan_out["wgt_state"] = wgt_state

            carry_next = {
                "hidden": hidden_next,
                "state_filt": (mean_state_next, var_state_next)
            }
            return carry_next, scan_out

        scan_init = {
            "hidden": hidden,
            "state_filt": (mean_state_init, var_state_init)
        }
        fargs = {
            "data_seq": data_seq,
            "x_meas": y_meas
        }

        final, scan_out = jax.lax.scan(f, scan_init, fargs)
        scan_out["state_filt"] = (
            jnp.concatenate([mean_state_init[None], scan_out["state_filt"][0].reshape(-1, self.n_state)]),
            jnp.concatenate([var_state_init[None], scan_out["state_filt"][1].reshape(-1, self.n_state, self.n_state)])
        )
        scan_out["state_pred"] = (
            scan_out["state_pred"][0].reshape(-1, self.n_state),
            scan_out["state_pred"][1].reshape(-1, self.n_state, self.n_state)
        )
        scan_out["wgt_state"] = scan_out["wgt_state"].reshape(-1, self.n_state, self.n_state)
        return scan_out


class RNN_theta(eqx.Module):
    hidden_size: int
    rnn_layers: list
    linear: eqx.Module
    
    def __init__(self, key, n_theta, n_inp):
        key, *subkey = jax.random.split(key, num=3)
        self.hidden_size = n_theta + (n_theta+1)*n_theta//2
        self.rnn_layers = [
            eqx.nn.GRUCell(n_inp, self.hidden_size, key=subkey[0]),
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[1])
        
    def __call__(self, y_meas):
        hidden = jnp.zeros((len(self.rnn_layers), self.hidden_size,))
        data_seq = y_meas
        for i in range(len(hidden)):
            def f(carry, inp):
                return self.rnn_layers[i](inp, carry), self.rnn_layers[i](inp, carry)
            final, data_seq = jax.lax.scan(f, hidden[i], data_seq)
        out = self.linear(final)
        return out


# NN for backward pass
class NN(eqx.Module):
    hidden_size: int
    out_size: int
    layers: list
    linear: eqx.Module
     
    def __init__(self, key, n_state):
        key, *subkey = jax.random.split(key, num=6)
        self.out_size = n_state + n_state*(n_state + 1)//2 + n_state*n_state
        n_inp = n_state*n_state + 2*n_state + n_state*(n_state + 1)//2
        self.hidden_size = n_inp * 2
        self.layers = [
            eqx.nn.Linear(n_inp, self.hidden_size, key=subkey[0], use_bias=True),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[1], use_bias=True),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[2], use_bias=True),
            jax.nn.relu,
            # eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[3]),
            # jax.nn.relu
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, self.out_size, key=subkey[4], use_bias=True)

    

    def __call__(self, input):
        
        for layer in self.layers:
            input = layer(input)
        
        out = self.linear(input)
        return out


class KalmanModel:
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
    by fitting the Kalman filtering parameters. Here, the model relies on the Kalman update to figure out the
    filtering estimates (i.e., it relies on using the fact that measurements are assumed to be Gaussian.) 
    This is to help the model to train faster and not converge to a poor local mode.

    Args:
        n_state (int): Dimension of the latent at each time point.
        n_theta (int): Number of parameters for inference.
        n_res (int): Resolution number.
        wgt_meas (jnp.array): Measurement selection matrix.
        var_meas (jnp.array): Measurement noise matrix.
    """

    def __init__(self, n_state, n_theta, n_res, wgt_meas, var_meas):
        self._n_state = n_state
        # self._obs_times = obs_times
        self._n_res = n_res
        self._wgt_meas = wgt_meas
        self._var_meas = var_meas
        # self._n_obs = len(obs_times)
        self._n_theta = n_theta

    def _rnn_input(self, y_meas, obs_times):
        # theta_rep = jnp.repeat(theta[None], self._n_obs-1, axis=0)
        y_meas_prev = y_meas[:-1]
        y_meas_next = y_meas[1:]
        y_meas_comb = jnp.hstack([y_meas_prev, y_meas_next])
        input = jnp.hstack([y_meas_comb, obs_times[1:, None]])
        return input

    def simulate_theta(self, key, params, y_meas):
        r"""
        Simulate $\theta$ using the variation distribution. Also compute $\log q(\theta)$.
        """

        # theta_mu = params["theta_mu"]
        theta_model = params["rnn_theta"]
        theta_out = theta_model(y_meas)
        theta_mu = theta_out[:self._n_theta]
        n_theta = len(theta_mu)
        # theta_std = jax.nn.softplus(params["theta_std"])
        random_normal = jax.random.normal(key, shape=(n_theta,))
        # theta = theta_mu + theta_std*random_normal
        # theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        theta_chol = theta_to_chol(theta_out[self._n_theta:], n_theta)
        theta = theta_mu + theta_chol.dot(random_normal)
        # theta_entpy = 0.5*n_theta*(1+jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.diag(theta_chol)))
        theta_entpy = -jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(theta, theta_mu, theta_chol.dot(theta_chol.T)))
        return theta, theta_entpy

    def simulate(self, key, params, y_meas, x_init, theta, obs_times):
        r"""
        Simulate $x$ using the variational distribution. Also compute $\log q(x \mid \theta)$.

        Args:
            key (random.key): PRNG key.
            params (dict): Dictionary to hold the neural network parameters and other parameters necessary to simulate $x$.
            y_meas (jnp.array): Observations.
            x_init (jnp.array): Initial guess of the latent.
            theta (jnp.array): SDE parameters.
            obs_times (jnp.array): Observation times.

        Returns:
            x (jnp.array): Latents at the SDE time points.
            logpdf (float): $\log q(x \mid \theta)$.
        """
        
        # y_meas = jax.vmap(lambda x, y: y - self._wgt_meas.dot(x))(x_init[::self._n_res], y_meas)
        data_seq = self._rnn_input(y_meas, obs_times)
        y_meas = jax.vmap(lambda x, y: y - self._wgt_meas.dot(x))(x_init[::self._n_res], y_meas)

        gru_model = params["gru"]
        # lam = params["lam"]
        mean_state_init = params["mean_init"]
        chol_state_init = theta_to_chol(params["chol_init"], self._n_state)
        var_state_init = chol_state_init.dot(chol_state_init.T)

        # do one update to use y_0
        mean_state_init, var_state_init = standard.update(
            mean_state_pred=mean_state_init,
            var_state_pred=var_state_init,
            x_meas=y_meas[0],
            mean_meas=jnp.zeros_like(y_meas[0]),
            wgt_meas=self._wgt_meas,
            var_meas=self._var_meas
        )

        gru_output = gru_model(data_seq, theta, mean_state_init, var_state_init, self._wgt_meas, self._var_meas, y_meas[1:])
        mean_state_filt, var_state_filt = gru_output["state_filt"]
        mean_state_pred, var_state_pred = gru_output["state_pred"]
        wgt_state = gru_output["wgt_state"]

        # mean_state_filt += x_init
        # mean_state_pred += x_init

        # NN model for the backward pass (conditional MVN)
        # nn_model = params["nn"]
        # lower_ind = jnp.tril_indices(self._n_state)
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
            # nn_input = jnp.concatenate([wgt_state_back.flatten(), mean_state_back, chol_back[lower_ind], x_state_next])
            # nn_output = nn_model(nn_input) 
            # wgt_state_back = lam * wgt_state_back + (1-lam) * nn_output[:self._n_state * self._n_state].reshape(self._n_state, self._n_state)
            # mean_state_back = lam * mean_state_back + (1-lam) * nn_output[self._n_state * self._n_state : self._n_state * self._n_state + self._n_state]
            # chol_curr = jnp.zeros((self._n_state, self._n_state))
            # chol_curr = chol_curr.at[lower_ind].set(nn_output[self._n_state * self._n_state + self._n_state:])
            # chol_curr = chol_curr.at[diag_ind].set(jax.nn.softplus(chol_curr[diag_ind]))
            # chol_curr = lam * chol_back + (1-lam) * chol_curr
            # var_state_curr = chol_curr.dot(chol_curr.T)
            mean_state_curr = wgt_state_back.dot(x_state_next) + mean_state_back
            chol_curr = chol_back
            var_state_curr = var_state_back
            x_state_curr = mean_state_curr + chol_curr.dot(random_normal) 
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(x_state_curr, mean_state_curr, var_state_curr)
            carry = {
                'x_state_next': x_state_curr,
                'x_neglogpdf': x_neglogpdf
            }
            return carry, carry
        
        # time N
        mean_state_N = mean_state_filt[-1]
        var_state_N = var_state_filt[-1]
        random_normals = jax.random.normal(key, shape=(len(mean_state_filt), self._n_state))
        chol_factor = jnp.linalg.cholesky(var_state_N)
        x_N = mean_state_N + chol_factor.dot(random_normals[-1])
        x_neglogpdf = -jax.scipy.stats.multivariate_normal.logpdf(x_N, mean_state_N, var_state_N)
        scan_init = {
            'x_state_next': x_N,
            'x_neglogpdf': x_neglogpdf
        }
        # scan arguments
        scan_kwargs = {
            'mean_state_filt': mean_state_filt[:-1],
            'var_state_filt': var_state_filt[:-1],
            'mean_state_pred': mean_state_pred,
            'var_state_pred': var_state_pred,
            'wgt_state': wgt_state,
            'random_normal': random_normals[:-1],
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
