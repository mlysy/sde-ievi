# %%
import os.path
import jax
import equinox as eqx
import optax 
import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
import jaxopt
import blackjax

import pfjax as pf
from pfjax.models.oratree_model import OraTreeModel as OTModel
import ievi.model_ievimixed as ievimixed
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

NUM_LOOPS = 100
n_particles = 400

# parameter values
phi1 = 195.
phi2 = 350.
sigma1 = 25.
sigma2 = 52.5
sigma = 0.08
N = 100
x0 = jnp.array([30.])
theta_true = jnp.array([phi1, sigma1, phi2, sigma2, sigma])
mean_idx = jnp.array([0, 2])
var_idx = jnp.array([1,3,4])
theta_names = ["phi1", "sigma1", "phi2", "sigma2", "sigma"]
theta_unc = theta_true.at[var_idx].set(jnp.log(theta_true[var_idx]))


dt_obs = 200
n_sim = 2
n_res = 2
n_obs = 16
n_state = 1
n_meas = 1
n_random = 2
obs_times = jnp.arange(n_obs)*dt_obs
sde_times = jnp.arange(n_res*(n_obs-1)+1)*dt_obs/n_res
ot_sim = OTModel(dt_obs, n_sim, N)
ot_model = OTModel(dt_obs, n_res, 1, bootstrap=False)

x_init = jnp.block([[jnp.zeros((n_sim-1, N))],
                    [jnp.array([30.0]*N)]])

### data generation------------------------------------------------------------------
key = jax.random.PRNGKey(0)
data_key, model_key, future_key = jax.random.split(key, num=3) # future key is for new models/datasets
data_iter_key = jax.random.split(data_key, num=NUM_LOOPS) # keys for datasets 
pfjax_key, ievi_key = jax.random.split(model_key, num=2) # keys for models

for i in range(NUM_LOOPS):
    key1, key2 = jax.random.split(data_iter_key[i])
    phi_sample = theta_true[mean_idx] + jax.random.normal(key1, (N, 2)) * theta_true[1::2]
    theta_sample = jnp.append(phi_sample, theta_true)
    y_meas, x_state = pf.simulate(ot_sim, key2, n_obs, x_init, theta_sample)
    # x_state = x_state.reshape(-1, N)[n_sim-1:][::int(n_sim/n_res)]
    y_meas = y_meas.T
    df = pd.DataFrame(y_meas)
    df['key_pair1'] = data_iter_key[i][0]
    df['key_pair2'] = data_iter_key[i][1]
    df.to_csv("data/data_{}.csv".format(i), index=False)
    # jnp.save("data/data_{}.npy".format(i), y_meas)
print("finished data gen")
    

# %%
def prior(theta):
    """Calculate the prior.
    
    Args: 
        theta: Hyperparameters
    
    Returns:
        l_prior: Logpdf of prior.
    """
    prior_mean_sd = 100
    prior_var_scale = 1/.01
    l_prior = jnp.sum(jsp.stats.norm.logpdf(theta[mean_idx], scale=prior_mean_sd))
    l_prior += jnp.sum(jsp.stats.expon.logpdf(1/theta[var_idx]/theta[var_idx], scale=prior_var_scale))
    return l_prior

# %%


### particle filtering + max & smooth-------------------------------------------------
def _pf_max(smooth_params, y_n, key):
    r""""
    Use pfjax to approximate p(y_n|smooth_params).

    """
    theta = smooth_params.at[2].set(jnp.exp(smooth_params[2]))
    pf_out = pf.particle_filter(model=ot_model, 
                                key=key,
                                y_meas=y_n, 
                                n_particles=n_particles,
                                resampler=pf.particle_resamplers.resample_mvn,
                                theta=theta)
    
    lp = pf_out['loglik']
    return -lp

def pf_max_all(key, smooth_params, y_meas):
    r"""
    Find the mode and inverse quadrature via the max-and-smooth method of p(y_meas | smooth_params).

    Parameters
    ----------
    smooth_params : jax.Array
        Parameters to optimize using the max-and-smooth method.
    y_meas : jax.Array
        Observations.

    Returns
    -------
    muN : jax.Array
        Mode of p(y_meas | smooth_params).
    SigmaN : jax.Array
        Inverse quadrature of p(y_meas | smooth_params).
    lp_tot : float
        Log-likelihood of p(y_meas | smooth_params).
    """

    solver = jaxopt.ScipyMinimize(
        fun=_pf_max,
        method="BFGS",
        # maxiter= 1000,
        # options={'gtol': 1e-6}
    )

    def vmap_fun(smooth_param, y_n):
        res = solver.run(smooth_param, y_n, key)
        mode = res.params
        var = res.state.hess_inv
        # var = jnp.linalg.inv(jax.hessian(_pf_max)(mode, y_n))
        nlp = res.state.fun_val
        return mode, var, nlp
    
    n_par = smooth_params.shape[1]
    muN = jnp.zeros((N, n_par))
    SigmaN = jnp.zeros((N, n_par, n_par))
    nlp_tot = 0.0
    for i in range(N):
        mu, Sigma, nlp = vmap_fun(smooth_params[i], y_meas[i])
        muN = muN.at[i].set(mu)
        SigmaN = SigmaN.at[i].set(Sigma)
        nlp_tot += nlp

    # mu, Sigma, nlp = jax.vmap(vmap_fun)(smooth_params, y_meas)
    return muN, SigmaN, -nlp_tot

def pf_max_joint(random_params, smooth_mu, smooth_Sigma, theta_unc):
    r"""
    Find the log-likelihood of p(random_params, theta_unc) using the max-and-smooth method.

    Parameters
    ----------
    random_params : jax.Array
        Random effects parameters.
    smooth_mu : jax.Array
        Mode found by the max-and-smooth method.
    smooth_Sigma : jax.Array
        Variance found by the max-and-smooth method.
    theta_unc : jax.Array
        Unconstrained hyperparameters.

    Returns
    -------
    lp : float
        Log-likelihood of p(random_params, theta_unc).
    """
    # max smooth approx
    theta = theta_unc.at[var_idx].set(jnp.exp(theta_unc[var_idx]))
    random_params = random_params.reshape(-1,2)
    N = len(random_params)
    smooth_params = jnp.hstack([random_params, jnp.repeat(theta_unc[-1], repeats=N, axis=0)[:, None]])
    lp_prior = prior(theta)
    lp_max = jnp.sum(jax.vmap(jsp.stats.multivariate_normal.logpdf)(smooth_params, smooth_mu, smooth_Sigma))
    lp_hyper = jnp.sum(jsp.stats.norm.logpdf(random_params, theta[mean_idx], theta[1::2]))
    return lp_prior + lp_max + lp_hyper

def pf_max_cond(random_params, smooth_mu, smooth_Sigma, theta_unc):
    r"""
    Find the log-likelihood of p(random_params | theta_unc) using the max-and-smooth method.

    Parameters
    ----------
    random_params : jax.Array
        Random effects parameters.
    smooth_mu : jax.Array
        Mode found by the max-and-smooth method.
    smooth_Sigma : jax.Array
        Variance found by the max-and-smooth method.
    theta_unc : jax.Array
        Unconstrained hyperparameters.

    Returns
    -------
    lp : float
        Log-likelihood of p(random_params | theta_unc).
    """
    n_params = len(random_params)
    grad = jax.grad(pf_max_joint)(jnp.zeros((n_params,)), smooth_mu, smooth_Sigma, theta_unc)
    hess_fun = jax.hessian(pf_max_joint)
    hess = hess_fun(jnp.zeros((n_params,)), smooth_mu, smooth_Sigma, theta_unc)
    cond_mu = -jnp.linalg.solve(hess, grad)
    # hess_at_mu = hess_fun(cond_mu, smooth_mu, smooth_Sigma, theta_unc)
    cond_var = -jnp.linalg.inv(hess)
    lp = jsp.stats.multivariate_normal.logpdf(random_params, cond_mu, cond_var)
    return lp

def pf_max_marg(theta_unc, smooth_mu, smooth_Sigma):
    r"""
    Find the log-likelihood of p(theta_unc) using the max-and-smooth method.

    Parameters
    ----------
    smooth_mu : jax.Array
        Mode found by the max-and-smooth method.
    smooth_Sigma : jax.Array
        Variance found by the max-and-smooth method.
    theta_unc : jax.Array
        Unconstrained hyperparameters.

    Returns
    -------
    lp : float
        Log-likelihood of p(theta_unc).
    """
    
    random_params = smooth_mu[:, :2].flatten()
    lp_joint = pf_max_joint(random_params, smooth_mu, smooth_Sigma, theta_unc)
    lp_cond = pf_max_cond(random_params, smooth_mu, smooth_Sigma, theta_unc)
    return lp_joint - lp_cond

def pf_max_negmarg(theta_unc, smooth_mu, smooth_Sigma):
    return -pf_max_marg(theta_unc, smooth_mu, smooth_Sigma)

def full_neglogpost(theta_unc, y_meas):
    theta = theta_unc.at[2*N+var_idx].set(jnp.exp(theta_unc[2*N+var_idx]))
    l_prior = prior(theta[2*N:])
    def vmap_fun(y_meas, theta_n):
        x_state = jnp.expand_dims(y_meas, axis=2)
        lp = pf.loglik_full(model=ot_model, 
                            y_meas=y_meas,
                            x_state=x_state,
                            theta=theta_n)
        
        return lp
    
    theta_x = theta[:2*N].reshape(-1,2)
    random_params = jnp.hstack([theta_x, jnp.repeat(theta[-1], repeats=N, axis=0)[:, None]])
    lp = jnp.sum(jax.vmap(vmap_fun)(y_meas, random_params))
    lp2 = jnp.sum(jsp.stats.norm.logpdf(theta_x, theta[2*N + mean_idx], theta[2*N + var_idx[:2]]))
    return -l_prior - lp - lp2 

for nl in range(NUM_LOOPS):
    # load data
    y_meas = jnp.array(
        pd.read_csv("data/data_{}.csv".format(nl)).iloc[:, :16])
    y_meas = jnp.expand_dims(y_meas, axis=2)
    
    # max and smooth
    solver = jaxopt.ScipyMinimize(
        fun=full_neglogpost,
        method="Newton-CG"
    )

    theta_init = jnp.append(195 * jnp.ones(2*N), jnp.array([190.5, 3.21, 345.5, 3.96, -2.5]))
    pf_res = solver.run(theta_init, y_meas)
    smooth_params = jnp.hstack([pf_res.params[:2*N].reshape(-1,2), jnp.repeat(pf_res.params[-1], repeats=N, axis=0)[:, None]])
    smooth_mu, smooth_Sigma, smooth_lp = pf_max_all(pfjax_key, smooth_params, y_meas)

    # mode/quadrature
    solver = jaxopt.ScipyMinimize(
        fun=pf_max_negmarg,
        method="BFGS"
    )
    pf_res = solver.run(pf_res.params[2*N:], smooth_mu, smooth_Sigma)
    pred_mu = pf_res.params
    pred_se =  jnp.sqrt(jnp.diag(pf_res.state.hess_inv))
    
    df = pd.DataFrame({
        "run_number": nl, 'theta':theta_names, 'mu':pred_mu, 'se':pred_se})
    if os.path.isfile('results/pfjax_oratree.csv'):
        df.to_csv('results/pfjax_oratree.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results/pfjax_oratree.csv', index=False)
    print("done run {} for pfjax".format(nl))



# ievi----------------------------------------------------------------------------
def logdensity(utheta, random_effect, x_state, y_meas):
    r"""
    Computes the joint density `p(x,y|theta)`.
    """
    dt_res = dt_obs/n_res
    theta = utheta.at[var_idx].set(jnp.exp(utheta[var_idx]))
    # theta = utheta.at[ind].set(jax.nn.softplus(utheta[ind]))
    theta_i = jnp.append(random_effect, theta[-1])
    # compute latent lpdf
    def state_lpdf(x_curr, x_prev, theta, dt):
        return jsp.stats.multivariate_normal.logpdf(
            x=x_curr,
            mean=x_prev + ot_model.drift(x_prev, theta) * dt,
            cov=ot_model.diff(x_prev, theta) * dt
        )
    state_lp = jnp.sum(jax.vmap(lambda xc, xp: 
        state_lpdf(xc, xp, theta_i, dt_res))(x_state[1:], x_state[:-1]))
    
    def meas_lpdf(y_curr, x_curr):
        return jsp.stats.norm.logpdf(
            x=y_curr,
            loc=x_curr,
            scale=jnp.array([1.])
        )
    meas_lp = jnp.sum(jax.vmap(meas_lpdf)(y_meas, x_state[::n_res]))
    random_lp = jnp.sum(jsp.stats.norm.logpdf(random_effect, theta[mean_idx], theta[var_idx[:2]]))
    return (state_lp + meas_lp + random_lp)


run_key, *nn_keys = jax.random.split(ievi_key, num=3)
n_theta = len(theta_unc)
lower_theta = jnp.tril_indices(n_theta)
random_ind = jnp.arange(4)
fixed_ind = jnp.array([4])

# condmvn RNN and NN
ievi_gru = ievimixed.RNN(nn_keys[0], n_state, 2*n_meas + 1)
ievi_random = ievimixed.NN_rand(nn_keys[1], len(random_ind) + 1, n_obs, len(random_ind)//2)


def inv_sp(x):
    return jnp.log(jnp.exp(x) - 1)

def trans(params):
    out = params.at[var_idx].set(jnp.exp(params[var_idx]))
    return out


def theta_to_std(theta_lower, n_theta):
    lower_ind = jnp.tril_indices(n_theta)
    theta_chol = jnp.zeros((n_theta, n_theta))
    theta_chol = theta_chol.at[lower_ind].set(theta_lower)
    return jax.nn.softplus(jnp.diag(theta_chol))

def batchloader(batch, y_meas, x_init):
    batch_size = N // n_batch
    return y_meas[batch*batch_size:(batch+1)*batch_size], x_init[batch*batch_size:(batch+1)*batch_size]


# %%
for nl in range(NUM_LOOPS):
    # load data
    y_meas = jnp.array(
        pd.read_csv("data/data_{}.csv".format(nl)).iloc[:, :16])
    x_meas = jax.vmap(jnp.interp, (None, None, 0))(sde_times, obs_times, y_meas)
    x_meas = jnp.expand_dims(x_meas, axis=2)
    y_meas = jnp.expand_dims(y_meas, axis=2)
    ievi_model = ievimixed.SmoothModel(n_state, random_ind, fixed_ind, obs_times, sde_times)
    theta_mu_init = jnp.array([190.5, 3.21, 345.5, 3.96, -2.5])
    ievi_params = {
        "gru": ievi_gru,
        "nn_random": ievi_random,
        "theta_mu": theta_mu_init,
        "theta_chol": jnp.diag(inv_sp(jnp.array([.1, .1, .1, .1, .1])))[lower_theta],
    }

    optim = optax.adam(0.001)
    
    @eqx.filter_value_and_grad
    def loss_fn(params, model, key,  y_meas, x_meas):
        n_sim = 1
        keys = jax.random.split(key, num=n_sim)
        def vmap_fun(key):
            (xsample, utheta, random_effect), gen_neglogpdf = model.simulate(key, params, y_meas, x_meas)
            rec_logpdf = jnp.sum(jax.vmap(logdensity,in_axes=[None, 0,0,0])(utheta, random_effect, xsample, y_meas))
            prior_logpdf = prior(utheta)
            loss =-(gen_neglogpdf + rec_logpdf + prior_logpdf)
            return loss
        loss = jnp.sum(jax.vmap(vmap_fun)(keys))
        return loss/n_sim
    

    @eqx.filter_jit
    def make_step(params, model, key,  y_meas, x_meas, opt_state):
        loss, grads = loss_fn(params, model, key,  y_meas, x_meas)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state

    
    opt_state = optim.init(eqx.filter(ievi_params, eqx.is_array))
    n_iter = 40000
    n_batch = 10
    min_loss = jnp.inf
    keys = jax.random.split(run_key, num=(n_iter, n_batch)) 
    for i in range(n_iter):
        for nb in range(n_batch):
            y_batch, x_batch = batchloader(nb, y_meas, x_meas)
            loss, ievi_params, opt_state = make_step(ievi_params, ievi_model, keys[i][nb], y_batch, x_batch, opt_state)
            loss = loss.item() * n_batch
            
            if jnp.isnan(loss):
                ievi_params = ievi_params_best
                opt_state = optim.init(eqx.filter(ievi_params, eqx.is_array))
                print("NAN LOSS")
            elif i < 2000:
                ievi_params_best = ievi_params
            elif loss < min_loss:
                min_loss = loss
                ievi_params_best = ievi_params
            # if i % 100 == 0 and nb == 0:
            #     mu_new = trans(ievi_params["theta_mu"])
            #     se_new = theta_to_std(ievi_params["theta_chol"], n_theta)
            #     print(f"step={i}, loss={loss}")
            #     print("theta_mu: ", mu_new)
            #     print("theta_std: ", se_new)
            if i == 30000:
                n_batch = 1

        
    pred_mu = ievi_params_best["theta_mu"]
    pred_se = theta_to_std(ievi_params_best["theta_chol"], n_theta)


    df = pd.DataFrame({
        'run_number':nl, 'theta':theta_names, 'mu':pred_mu, 'se':pred_se})
    if os.path.isfile('results/ievi_oratree.csv'):
        df.to_csv('results/ievi_oratree.csv', mode='a', header=False, index=False) 
    else:
        df.to_csv('results/ievi_oratree.csv', index=False)
    
    print("done run {} for ievi".format(nl))
    (ievi_x, utheta, random_effect), neglogpdf = ievi_model.simulate(key, ievi_params_best, y_meas, x_meas)
    ievi_model._save_pars(ievi_params_best, y_meas)
    
    mcmc_key1, mcmc_key2 = jax.random.split(ievi_key)

    keys = jax.random.split(mcmc_key1, N)
    def logpost_fn(utheta, random_effect):
        fixed_effect = jnp.exp(utheta[-1])
        random_fixed = jax.vmap(jnp.append, in_axes=[0, None])(random_effect, fixed_effect)
        x_state, x_neglogpdf = ievi_model._sim_trained(keys, random_fixed, x_meas)
        rec_logpdf = jnp.sum(jax.vmap(logdensity,in_axes=[None, 0,0, 0])(utheta, random_effect, x_state, y_meas))
        prior_logpdf = prior(utheta)
        loss = rec_logpdf + prior_logpdf - jnp.sum(x_neglogpdf)
        return loss

    logpost = jax.jit(lambda x: logpost_fn(**x))
    utheta_init = ievi_params_best["theta_mu"]
    random_init = random_effect
    upars_init = {"utheta":utheta_init, "random_effect": random_init}
    
    hmc_key1, hmc_key2 = jax.random.split(mcmc_key2)
    # Stan window adaptation algorithm to start with reasonable parameters
    num_steps = 10
    warmup = blackjax.window_adaptation(
        blackjax.hmc, logpost, num_integration_steps=num_steps)

    (initial_state, parameters), _ = warmup.run(hmc_key1, upars_init)
    kernel = blackjax.hmc(logpost, **parameters).step
    # standard in blackjax to write an inference loop for sampling
    def inference_loop(key, kernel, initial_state, n_samples):

        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(key, n_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states
    
    n_samples = 10000
    uode_sample = inference_loop(hmc_key2, kernel, initial_state, n_samples)
    uode_sample2 = uode_sample.position["utheta"]
    
    pred_mu = jnp.mean(uode_sample2, axis=0)
    pred_se = jnp.std(uode_sample2, axis=0)
    df = pd.DataFrame({
            'run_number':nl, 'theta':theta_names, 'mu':pred_mu, 'se':pred_se})
    if os.path.isfile('results/ievimcmc_oratree.csv'):
        df.to_csv('results/ievimcmc_oratree.csv', mode='a', header=False, index=False) 
    else:
        df.to_csv('results/ievimcmc_oratree.csv', index=False)
    
    print("done run {} for mcmc".format(nl))
