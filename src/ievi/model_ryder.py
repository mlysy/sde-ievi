import jax
import jax.numpy as jnp
import equinox as eqx
from svi.utils import theta_to_chol


# Ruder NN model
class RyderNN(eqx.Module):
    hidden_size: int
    out_size: int
    layers: list
    linear: eqx.Module

    def __init__(self, key, n_state, n_inp):
        key, *subkey = jax.random.split(key, num=6)
        self.out_size = n_state + n_state * (n_state + 1) // 2
        self.hidden_size = 50
        self.layers = [
            eqx.nn.Linear(n_inp, self.hidden_size, key=subkey[0]),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[1]),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[2]),
            jax.nn.relu,
            eqx.nn.Linear(self.hidden_size, self.hidden_size, key=subkey[3]),
            jax.nn.relu,
        ]
        self.linear = eqx.nn.Linear(self.hidden_size, self.out_size, key=subkey[4])

    def __call__(self, input):

        for layer in self.layers:
            input = layer(input)

        out = self.linear(input)
        return out


class RyderModel:
    r"""
    This module implement the RNN method of Ryder at al. (2018)

    RNN takes on the features:
    \begin{itemize}
        \item $\theta$
        \item $x_n$
        \item $t'_m - t_n$
        \item $t'_m$
        \item $y_m$
    \end{itemize}

    where $t_0, ... t_N$ is the SDE time grid and $t'_0, ..., t'_M$ is the observation time grid.
    Here $t'_m$ is the next observation time greater than $t_n$.

    The RNN returns $\alpha, \beta^{1/2}$ where simulation is done via $x_n = x_{n-1} + \alpha \Delta t + \beta^{1/2} \sqrt \Delta t \epsilon$.

    Parameters
    ----------
    n_state : int
        Dimension of the latent at each time point.
    obs_times : jax.Array
        The observation time points.
    sde_times : jax.Array
        The discretization time points used to sample the latents.
    obs_mat : jax.Array
        The fixed observation matrix assumed in Ryder.
    restrict : bool
        Flag for determining whether the simulated $x$ needs to be positive.
    """

    def __init__(self, n_state, obs_times, sde_times, obs_mat, restrict=False):
        self._n_state = n_state
        self._obs_times = obs_times
        self._sde_times = sde_times
        self._dt = sde_times[1] - sde_times[0]
        self._obs_mat = obs_mat
        self._restrict = restrict

    def _nn_input(self, theta, y_meas):
        theta_rep = jnp.repeat(theta[None], len(self._sde_times) - 1, axis=0)
        obs_ind = jnp.searchsorted(self._obs_times, self._sde_times[:-1], side="right")
        time_next = self._obs_times[obs_ind]
        time_diff = time_next - self._sde_times[:-1]
        y_meas_next = y_meas[obs_ind]
        input = jnp.concatenate(
            [theta_rep, self._sde_times[:-1, None], time_diff[:, None], y_meas_next],
            axis=1,
        )
        return input

    def simulate_theta(self, key, params):
        r"""
        Simulate $\theta$ using the variation distribution. Also compute $\log q(\theta)$.
        """

        theta_mu = params["theta_mu"]
        n_theta = len(theta_mu)
        random_normal = jax.random.normal(key, shape=(n_theta,))
        theta_chol = theta_to_chol(params["theta_chol"], n_theta)
        theta = theta_mu + theta_chol.dot(random_normal)
        theta_entpy = -jax.scipy.stats.multivariate_normal.logpdf(
            theta, theta_mu, theta_chol.dot(theta_chol.T)
        )
        return theta, theta_entpy

    def simulate_theta_rnn(self, key, params, y_meas, n_theta):
        r"""
        Simulate $\theta$ using the variation distribution assuming a RNN. Also compute $\log q(\theta)$.
        """
        theta_model = params["rnn_theta"]
        theta_out = theta_model(y_meas)
        theta_mu = theta_out[:n_theta]
        random_normal = jax.random.normal(key, shape=(n_theta,))
        theta_chol = theta_to_chol(theta_out[n_theta:], n_theta)
        theta = theta_mu + theta_chol.dot(random_normal)
        theta_entpy = -jnp.sum(
            jax.scipy.stats.multivariate_normal.logpdf(
                theta, theta_mu, theta_chol.dot(theta_chol.T)
            )
        )
        return theta, theta_entpy

    def simulate(self, key, params, y_meas, x_init, theta):
        r"""
        Simulate $x$ using Ryder and compute $\log q(x \mid \theta)$.

        Parameters
        ----------
        key : jax.PRNGKey
            PRNG key.
        params : dict
            Dictionary holding the neural network parameters and other parameters necessary to simulate $\theta$ and $x$.
        y_meas : jax.Array
            Observations.
        x_init : jax.Array
            Initial value for the latent SDE.
        theta : jax.Array
            SDE parameters.

        Returns
        -------
        x : jax.Array
            Latents at the SDE time points.
        logpdf : float
            $\log q(x \mid \theta)$.
        """

        n_theta = len(theta)
        nn_model = params["nn"]
        nn_input = self._nn_input(theta, y_meas)
        lower_ind = jnp.tril_indices(self._n_state)
        diag_ind = jnp.diag_indices(self._n_state)

        def sim_step(carry, args):
            nn_input = args["nn_input"]
            random_normal = args["random_normal"]
            x_prev = carry["x_curr"]
            x_neglogpdf = carry["x_neglogpdf"]
            y_meas = nn_input[n_theta + 2 :]
            nn_input_curr = jnp.concatenate(
                [nn_input[: n_theta + 2], x_prev, y_meas - self._obs_mat.dot(x_prev)]
            )
            nn_output = nn_model(nn_input_curr)
            alpha = nn_output[: self._n_state]
            beta_lower = jnp.zeros((self._n_state, self._n_state))
            beta_lower = beta_lower.at[lower_ind].set(nn_output[self._n_state :])
            beta_lower = beta_lower.at[diag_ind].set(
                jax.nn.softplus(beta_lower[diag_ind])
            ) + 1e-3 * jnp.eye(self._n_state)
            beta = beta_lower.dot(beta_lower.T)
            beta_lower = jnp.linalg.cholesky(beta)
            x_curr_untrans = (
                x_prev
                + alpha * self._dt
                + beta_lower.dot(random_normal) * jnp.sqrt(self._dt)
            )
            if self._restrict:
                x_curr = jax.nn.softplus(
                    x_curr_untrans
                )  # necessary if x > 0 is a restriction
            else:
                x_curr = x_curr_untrans
            x_neglogpdf -= jax.scipy.stats.multivariate_normal.logpdf(
                x_curr_untrans, x_prev + alpha * self._dt, beta * self._dt
            )
            carry = {"x_curr": x_curr, "x_neglogpdf": x_neglogpdf}
            return carry, carry

        scan_init = {"x_curr": x_init, "x_neglogpdf": 0.0}
        sde_len = len(self._sde_times)
        random_normals = jax.random.normal(key, shape=(sde_len - 1, self._n_state))
        # scan arguments
        scan_kwargs = {"nn_input": nn_input, "random_normal": random_normals}

        last_out, stack_out = jax.lax.scan(sim_step, scan_init, scan_kwargs)
        xs = jnp.concatenate([x_init[None], stack_out["x_curr"]])
        # calculate -E[log q(x, theta|phi_full)]
        # break down to - E[log q(theta)] - E[log q(x|theta)]
        x_neglogpdf = last_out["x_neglogpdf"]
        return xs, x_neglogpdf
