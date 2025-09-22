import jax
import jax.numpy as jnp

def theta_to_chol(theta_lower, n_theta):
    lower_ind = jnp.tril_indices(n_theta)
    diag_ind = jnp.diag_indices(n_theta)
    theta_chol = jnp.zeros((n_theta, n_theta))
    theta_chol = theta_chol.at[lower_ind].set(theta_lower)
    theta_chol = theta_chol.at[diag_ind].set(jax.nn.softplus(jnp.diag(theta_chol)))
    return theta_chol

def chol_to_var(chol_mat):
    var_mat = chol_mat.dot(chol_mat.T)
    return var_mat
