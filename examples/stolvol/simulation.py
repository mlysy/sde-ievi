# %%
import os
import numpy as np
import jax
import equinox as eqx
import optax 
import blackjax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import seaborn as sns
import jaxopt
import pandas as pd


import pfjax as pf
from pfjax.models.stovol_model import StoVolModel
import ievi.model_ievikalman as rk
import ievi.model_ryder as mr
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


NUM_LOOPS = 100
n_particles = 400
                          
# parameter values
alpha = 0.04
gamma = 3.0
mu = -1.3
sigma = 1.0
rho = -0.7
x0 = jnp.array([jnp.log(1000), .1])
theta = jnp.array([alpha, gamma, mu, sigma, rho])
theta_true = jnp.array([alpha, jnp.log(gamma), mu, jnp.log(sigma), jnp.log((rho+1)/(1-rho))])
theta_names = [r"$\alpha$", r"$\gamma$", r"$\mu$", r"$\sigma$", r"$\rho$"]
n_theta = len(theta_true)

dt_obs = 10/252
n_res = 2
n_sim = 128
n_obs = 1500
n_state = 2
n_meas = 1
obs_times = jnp.arange(n_obs)*dt_obs
sde_times = jnp.arange(n_res*(n_obs-1)+1)*dt_obs/n_res
sv_sim = StoVolModel(dt_obs, n_sim)
sv_model = StoVolModel(dt_obs, n_res, bootstrap=False)

x_init0 = jnp.block([[jnp.zeros((n_sim-1, 2))],
                    [x0]])

def trans_reg(utheta):
    # convert log gamma to gamma
    utheta = utheta.at[1].set(jnp.exp(utheta[1]))
    # convert log sigma to sigma
    utheta = utheta.at[3].set(jnp.exp(utheta[3]))
    # convert rho back to reg scale
    utheta = utheta.at[4].set((jnp.exp(utheta[4]) - 1)/(jnp.exp(utheta[4]) + 1))
    return utheta

def trans_unc(theta):
    return jnp.array([theta[0], jnp.log(theta[1]), theta[2], jnp.log(theta[3]), jnp.log((theta[4]+1)/(1-theta[4]))])

### data generation------------------------------------------------------------------
key = jax.random.PRNGKey(0)
data_key, model_key, future_key = jax.random.split(key, num=3) # future key is for new models/datasets
data_iter_key = jax.random.split(data_key, num=NUM_LOOPS) # keys for datasets 
ievi_key, ryder_key, pfjax_key, mcmc_key = jax.random.split(model_key, num=4) # keys for models


for i in range(NUM_LOOPS):
    y_meas, x_state = pf.simulate(sv_sim, data_iter_key[i], n_obs, x_init0, theta)
    x_state = x_state.reshape(-1, 2)[n_sim-1:][::int(n_sim/n_res)]
    df = pd.DataFrame({'noise':y_meas[:, 0], 'no_noise': x_state[::n_res, 0],
                      'key_pair1': data_iter_key[i][0], 'key_pair2': data_iter_key[i][1]})
    df.to_csv("data/data_{}.csv".format(i), index=False)




# %%
def logdensity(theta, x_state, y_meas, dt_obs, n_res):
    r"""
    Computes the joint density p(x,y|theta).
    """
    dt_res = dt_obs/n_res
    # convert back to reg scale
    # compute latent lpdf
    def state_lpdf(x_curr, x_prev, theta, dt):
        return jsp.stats.multivariate_normal.logpdf(
            x=x_curr,
            mean=x_prev + sv_model.drift(x_prev, theta) * dt,
            cov=sv_model.diff(x_prev, theta) * dt
        )
    state_lp = jnp.sum(jax.vmap(lambda xc, xp: 
        state_lpdf(xc, xp, theta, dt_res))(x_state[1:], x_state[:-1]))
    
    def meas_lpdf(y_curr, x_curr):
        return jsp.stats.norm.logpdf(
            x=y_curr,
            loc=x_curr,
            scale=1e-3
        )
    meas_lp = jnp.sum(jax.vmap(meas_lpdf)(y_meas, x_state[::n_res, 0]))
    return (state_lp + meas_lp)

def theta_to_std(theta_lower, n_theta):
    lower_ind = jnp.tril_indices(n_theta)
    theta_chol = jnp.zeros((n_theta, n_theta))
    theta_chol = theta_chol.at[lower_ind].set(theta_lower)
    return jax.nn.softplus(jnp.diag(theta_chol))


run_key, weight_key, *nn_keys = jax.random.split(ievi_key, num=6)
# Smooth RNN and NN setup
gru_model = rk.RNN(nn_keys[0], n_state, 2*n_meas + 1 + n_state + (n_state + 1)*n_state//2, n_theta, n_res)
lin_model = mr.RyderNN(nn_keys[1], n_state, n_state + n_meas + 2 + n_theta)
theta_model = rk.RNN_theta(nn_keys[2], n_theta, n_meas)
theta_model2 = rk.RNN_theta(nn_keys[3], n_theta, n_meas)
lower_theta = jnp.tril_indices(n_theta)
lower_state = jnp.tril_indices(n_state)
wgt_meas = jnp.array([[1., 0.]])
var_meas = jnp.array([[1e-3]])**2


# %%
### ievi ------------------------------------------------------------------------------------------
for nl in range(NUM_LOOPS):
    y_meas = jnp.array(
        pd.read_csv("data/data_{}.csv".format(nl)).iloc[:, :1])
    
    
    x_state2 = jnp.ones((len(sde_times), n_state)) * x0
    x_init = x_state2.at[:, 0].set(jnp.interp(sde_times, obs_times, y_meas[:, 0]))
    rk_model = rk.KalmanModel(n_state, n_theta, n_res, wgt_meas, var_meas)

    rk_params = {
        "gru": gru_model,
        "rnn_theta": theta_model,
        "x_init": x_init,
        "mean_init": jnp.zeros((n_state)), 
        "chol_init": jnp.eye(n_state)[lower_state] * -2
    }


    # training for full VI
    @eqx.filter_value_and_grad
    def loss_fnp(params, y_meas, dt_obs, n_res):
        n_sim = 1
        def vmap_fun(s):
            utheta = theta_true
            x_init = params["x_init"]
            theta = trans_reg(utheta)
            logdens = logdensity(theta, x_init, y_meas, dt_obs, n_res)
            return -(logdens )
        loss = jnp.sum(jax.vmap(vmap_fun)(jnp.arange(n_sim)))
        return loss/n_sim


    @eqx.filter_jit
    def make_stepp(params, y_meas, dt_obs, n_res, opt_state):
        loss, grads = loss_fnp(params,  y_meas, dt_obs, n_res)
        # gradtrans = optax.clip_by_global_norm(14e6)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state

    # learn proxy
    learning_rate = 0.01
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(rk_params, eqx.is_array))

    n_iter = 1000
    for i in range(n_iter):
        loss, rk_params, opt_state = make_stepp(rk_params, y_meas, dt_obs, n_res, opt_state)
        # print(jnp.max(jnp.abs(rk_params["x_init"]-x_state)))

    

    # # training for full VI
    @eqx.filter_value_and_grad
    def loss_fnf(params, model, key, y_meas, dt_obs, n_res, obs_times, x_init):
        n_sim = 1
        keys = jax.random.split(key, num=(n_sim, 2))
        def vmap_fun(key):
            utheta, t_entropy = model.simulate_theta(key[0], params, y_meas)
            theta = trans_reg(utheta)
            xsample, entropy = model.simulate(key[1], params, y_meas, x_init, theta, obs_times)
            logdens = logdensity(theta, xsample, y_meas, dt_obs, n_res)
            return -(logdens + entropy + t_entropy)
        loss = jnp.sum(jax.vmap(vmap_fun)(keys))
        return loss/n_sim

    @eqx.filter_jit
    def make_stepf(params, model, key,  y_meas, dt_obs, n_res, obs_times, x_init, opt_state):
        loss, grads = loss_fnf(params, model, key,  y_meas, dt_obs, n_res, obs_times, x_init)
        # gradtrans = optax.clip_by_global_norm(1e6)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state

    scheduler = optax.piecewise_constant_schedule(init_value=0.005,
                                            boundaries_and_scales={2000:.2})
    optim = optax.adam(scheduler)
    opt_state = optim.init(eqx.filter(rk_params, eqx.is_array))

    n_iter = 4000
    n_batch = 300
    keys = jax.random.split(run_key, num=(n_iter, n_obs//n_batch))
    min_loss = 1e100

    for i in range(n_iter):
        for j in range(n_obs//n_batch):
            y_j = y_meas[n_batch*j:n_batch*(j+1)]
            obs_j = obs_times[n_batch*j:n_batch*(j+1)]
            x_j = rk_params["x_init"][n_res*n_batch*j:n_res*n_batch*(j+1)-1]
            loss, rk_params, opt_state = make_stepf(rk_params, rk_model, keys[i][j], y_j, dt_obs, n_res, obs_j, x_j, opt_state)
            loss = loss.item()
            if loss < min_loss:
                rk_params_best = rk_params
                min_loss = loss
            if jnp.isnan(loss):
                rk_params = rk_params_best
                opt_state = optim.init(eqx.filter(rk_params, eqx.is_array))
                print("NAN LOSS")
        # if i % 100 == 0:
        #     print(f"step={i}, loss={loss}")
        #     theta_model = rk_params["rnn_theta"]
        #     theta_out = theta_model(y_j)
        #     print(trans_reg(theta_out[:n_theta]))
        #     print(theta_to_std(theta_out[n_theta:], n_theta))
        # if i % 2000 == 0:
        #     learning_rate = 0.005
        #     optim = optax.adam(learning_rate)
        #     # n_batch = 1500
        #     # keys = jax.random.split(key, num=(n_iter, n_obs//n_batch))
        #     opt_state = optim.init(eqx.filter(rk_params, eqx.is_array))
        
    theta_model = rk_params_best["rnn_theta"]
    theta_out = theta_model(y_meas)
    # print(trans_reg(theta_out[:n_theta]))
    # print(theta_to_std(theta_out[n_theta:], n_theta))
    df = pd.DataFrame({'run_number':nl, 'theta':theta_names, 'mu':theta_out[:n_theta], 'se':theta_to_std(theta_out[n_theta:], n_theta)})
    if os.path.isfile('results/ievi_stolvol.csv'):
        df.to_csv('results/ievi_stolvol.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results/ievi_stolvol.csv', index=False)
    print("done run {}".format(nl))


# # %%
# ### Ryder ---------------------------------------------------------------------------------------------
for nl in range( NUM_LOOPS):
    y_meas = jnp.array(
        pd.read_csv("data/data_{}.csv".format(nl)).iloc[:, :1])
    
    ryder_model = mr.RyderModel(n_state, obs_times, sde_times, jnp.array([1., 0.]))
    ryder_params = {
        "nn": lin_model,
        "rnn_theta": theta_model2
    }
    
    # training for full VI
    @eqx.filter_value_and_grad
    def loss_fnf(params, model, key, y_meas, dt_obs, n_res):
        n_sim = 1
        keys = jax.random.split(key, num=(n_sim, 2))
        def vmap_fun(key):
            utheta, t_entropy = model.simulate_theta_rnn(key[0], params, y_meas, n_theta)
            theta = trans_reg(utheta)
            xsample, entropy = model.simulate(key[1], params, y_meas, x0, theta)
            logdens = logdensity(theta, xsample, y_meas, dt_obs, n_res)
            return -(logdens + entropy + t_entropy)
        loss = jnp.sum(jax.vmap(vmap_fun)(keys))
        return loss/n_sim


    @eqx.filter_jit
    def make_stepf(params, model, key,  y_meas, dt_obs, n_res, opt_state):
        loss, grads = loss_fnf(params, model, key,  y_meas, dt_obs, n_res)
        # gradtrans = optax.clip_by_global_norm(1e6)
        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return loss, params, opt_state
    
    learning_rate = 0.005
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(ryder_params, eqx.is_array))

    n_iter = 3000
    keys = jax.random.split(run_key, num=(n_iter, ))
    min_loss = 1e100

    for i in range(n_iter):
        loss, ryder_params, opt_state = make_stepf(ryder_params, ryder_model, keys[i], y_meas, dt_obs, n_res, opt_state)
        loss = loss.item()
        if loss < min_loss:
            ryder_params_best = ryder_params
            min_loss = loss
        if jnp.isnan(loss):
            ryder_params = ryder_params_best
            break
        # if i % 100 == 0:
        #     print(f"step={i}, loss={loss}")
        #     theta_model = ryder_params["rnn_theta"]
        #     theta_out = theta_model(y_meas)
        #     print(trans_reg(theta_out[:n_theta]))
        #     print(theta_to_std(theta_out[n_theta:], n_theta))

    theta_model = ryder_params_best["rnn_theta"]
    theta_out = theta_model(y_meas)
    df = pd.DataFrame({'run_number':nl, 'theta':theta_names, 'mu':theta_out[:n_theta], 'se':theta_to_std(theta_out[n_theta:], n_theta)})
    if os.path.isfile('results/ryder_stolvol.csv'):
        df.to_csv('results/ryder_stolvol.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results/ryder_stolvol.csv', index=False)
    print("done run {}".format(nl))


# HMC ------------------------------------------------------------------------------------------------

for nl in range(NUM_LOOPS):
    y_meas = jnp.array(pd.read_csv("data/data_{}.csv".format(nl))["noise"]).reshape(-1, 1)

    def logpost_fn(utheta, x_state):
        r"""
        Computes the joint density p(x,y|theta).
        """
        dt_res = dt_obs/n_res
        # convert back to reg scale
        theta = trans_reg(utheta)
        # compute latent lpdf
        def state_lpdf(x_curr, x_prev, theta, dt):
            return jsp.stats.multivariate_normal.logpdf(
                x=x_curr,
                mean=x_prev + sv_model.drift(x_prev, theta) * dt,
                cov=sv_model.diff(x_prev, theta) * dt
            )
        state_lp = jnp.sum(jax.vmap(lambda xc, xp: 
            state_lpdf(xc, xp, theta, dt_res))(x_state[1:], x_state[:-1]))
        
        def meas_lpdf(y_curr, x_curr):
            return jsp.stats.norm.logpdf(
                x=y_curr,
                loc=x_curr,
                scale=1e-3 
            )
        meas_lp = jnp.sum(jax.vmap(meas_lpdf)(y_meas, x_state[::n_res, 0]))
        return state_lp + meas_lp

    x_state2 = 0.1 * jnp.ones((len(sde_times), n_state))
    x_init = x_state2.at[:, 0].set(jnp.interp(sde_times, obs_times, y_meas[:, 0]))

    logpost = lambda x: logpost_fn(**x)
    utheta_init = jnp.array(theta_true)

    upars_init = {"utheta":utheta_init, "x_state": x_init}
    key, *subkeys = jax.random.split(key, num=3)
    # Stan window adaptation algorithm to start with reasonable parameters
    num_steps = 200
    warmup = blackjax.window_adaptation(
        blackjax.hmc, logpost, num_integration_steps=num_steps)

    (initial_state, parameters), _ = warmup.run(subkeys[0], upars_init)
    kernel = blackjax.hmc(logpost, **parameters).step
    # standard in blackjax to write an inference loop for sampling
    def inference_loop(key, kernel, initial_state, n_samples):

        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(key, n_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states
    
    n_samples = 1000
    fmcmc_sample = inference_loop(subkeys[1], kernel, initial_state, n_samples)
    fmcmc_sample2 = fmcmc_sample.position["utheta"]
    fmcmc_pars = jax.vmap(trans_reg)(fmcmc_sample2)
    # np.save("saves/fmcmc_sv_seed0_nres{}".format(n_res), fmcmc_pars)

    
    theta_names = [r"$\alpha$", r"$\gamma$", r"$\mu$", r"$\sigma$", r"$\rho$"]
    df = pd.DataFrame({'run_number':nl, 'theta':theta_names, 'mu':jnp.mean(fmcmc_sample2,axis=0), 'se':jnp.std(fmcmc_sample2,axis=0)})
    if os.path.isfile('results/mcmc_stolvol.csv'):
        df.to_csv('results/mcmc_stolvol.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results/mcmc_stolvol.csv', index=False)
    
    print("done run:{}".format(nl))

### Particle Filtering ------------------------------------------------------------------------------

for nl in range(NUM_LOOPS):
    y_meas = jnp.array(pd.read_csv("data/data_{}.csv".format(nl))["noise"]).reshape(-1, 1)
   

    def pf_negloglik(utheta):
        theta = trans_reg(utheta)
        pf_out = pf.particle_filter(model=sv_model, 
                                    key=key,
                                    y_meas=y_meas, 
                                    n_particles=n_particles,
                                    resampler=pf.particle_resamplers.resample_mvn,
                                    theta=theta)
        return -pf_out['loglik']

    solver = jaxopt.ScipyMinimize(
        fun=pf_negloglik,
        method="BFGS",
        jit=True,
        maxiter=5000,
        options={'gtol': 1e-6, 'disp': True}
    )
    theta_init = jnp.array([alpha, jnp.log(gamma), mu, jnp.log(sigma), jnp.log((rho+1)/(1-rho))])
    # theta_init = pf_res.params
    pf_res = solver.run(theta_init)
    
    theta_names = [r"$\alpha$", r"$\gamma$", r"$\mu$", r"$\sigma$", r"$\rho$"]
    df = pd.DataFrame({'run_number':nl, 'theta':theta_names, 'mu':pf_res.params, 'se':jnp.sqrt(jnp.diag(pf_res.state.hess_inv))})
    if os.path.isfile('results/pf_stolvol.csv'):
        df.to_csv('results/pf_stolvol.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results/pf_stolvol.csv', index=False)
    
    print("done run:{}".format(nl))
