# %%
import os.path
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
import jaxopt
import equinox as eqx
import optax
jax.config.update('jax_platform_name', 'cpu') # faster on cpu for RNN models
jax.config.update("jax_enable_x64", True)


import pfjax as pf
from pfjax.models.rydlotvolreg_model import RyderLotVolModel as RegLVModel
from pfjax.models.rydlotvol_model import RyderLotVolModel as LVModel
import ievi.model_ievi as sde_cond
import ievi.model_ryder as mr



# %%

NUM_LOOPS = 100
n_particles = 800

# parameter values
alpha = 0.5
beta = 0.0025
gamma = 0.3
tau_h = 1
tau_l = 1
x0 = jnp.array([71., 79.])
theta = jnp.array([alpha, beta, gamma])
theta_unc = jnp.log(theta)
theta_names = ["alpha", "beta", "gamma"]

dt_obs = 10
n_sim = 1000
n_res = 100
n_obs = 5
n_state = 2
n_meas = 2
obs_times = jnp.arange(n_obs)*dt_obs
sde_times = jnp.arange(n_res*(n_obs-1)+1)*dt_obs/n_res
lv_model = LVModel(dt_obs, n_res, bootstrap=False)
reglv_model = RegLVModel(dt_obs, n_res)
lv_sim = LVModel(dt_obs, n_sim)
x_init = jnp.block([[jnp.zeros((n_sim-1, 2))],
                    [jnp.log(x0)]])

### data generation------------------------------------------------------------------
key = jax.random.PRNGKey(0)
data_key, model_key, future_key = jax.random.split(key, num=3) # future key is for new models/datasets
data_iter_key = jax.random.split(data_key, num=NUM_LOOPS) # keys for datasets 
pfjax_key, ryder_key, ievi_key = jax.random.split(model_key, num=3) # keys for models



# # %%
for i in range(NUM_LOOPS):
    y_meas, x_state = pf.simulate(lv_sim, data_iter_key[i], n_obs, x_init, theta)
    x_state = x_state.reshape(-1, 2)[n_sim-1:][::int(n_sim/n_res)]
    df = pd.DataFrame(y_meas)
    df['key_pair1'] = data_iter_key[i][0]
    df['key_pair2'] = data_iter_key[i][1]
    df.to_csv("data/data_{}.csv".format(i), index=False)
    # jnp.save("data/data_{}.npy".format(i), y_meas)
print("finished data gen")

# %%
### pfjax----------------------------------------------------------------------------- 

def pf_negloglik(utheta, y_meas):
    theta = jnp.exp(utheta)
    pf_out = pf.particle_filter(model=lv_model, 
                                key=pfjax_key,
                                y_meas=y_meas, 
                                n_particles=n_particles,
                                resampler=pf.particle_resamplers.resample_mvn,
                                theta=theta)
    return -pf_out['loglik']


for i in range(NUM_LOOPS):
    y_meas = jnp.array(
        pd.read_csv("data/data_{}.csv".format(i)).iloc[:, :2])
    solver = jaxopt.ScipyMinimize(
        fun=pf_negloglik,
        method="BFGS",
        jit=True,
        # maxiter=5000,
        # options={'gtol': 1e-6, 'disp': False}
    )

    pf_res = solver.run(theta_unc, y_meas)
    pred_mu = pf_res.params
    pred_se =  jnp.sqrt(jnp.diag(pf_res.state.hess_inv))
    
    df = pd.DataFrame({
        'run_number':i,
        'theta':theta_names, 'mu':pred_mu, 'se':pred_se})
    if os.path.isfile('results/pfjax_loglotvol.csv'):
        df.to_csv('results/pfjax_loglotvol.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results/pfjax_loglotvol.csv', index=False)
    print("done run {} for pfjax".format(i))

# %%

### ievi----------------------------------------------------------------------------

def logdensity(logtheta, x_state, y_meas, dt_obs, n_res):
    r"""
    Computes the joint density `p(x,y|theta)`.
    """
    dt_res = dt_obs/n_res
    theta = jnp.exp(logtheta)
    # compute latent lpdf
    def state_lpdf(x_curr, x_prev, theta, dt):
        return jsp.stats.multivariate_normal.logpdf(
            x=x_curr,
            mean=x_prev + lv_model.drift(x_prev, theta) * dt,
            cov=lv_model.diff(x_prev, theta) * dt
        )
    state_lp = jnp.sum(jax.vmap(lambda xc, xp: 
        state_lpdf(xc, xp, theta, dt_res))(x_state[1:], x_state[:-1]))


    # compute observation lpdf
    def meas_lpdf(y_curr, x_curr):
        return jsp.stats.norm.logpdf(
            x=y_curr,
            loc=jnp.exp(x_curr),
            scale=jnp.array([tau_h, tau_l])
            # scale=jnp.array([tau_l])
        )
    meas_lp = jnp.sum(jax.vmap(meas_lpdf)(y_meas, x_state[::n_res]))
    return state_lp + meas_lp


run_key, *nn_keys = jax.random.split(ievi_key, num=3)
n_theta = len(theta_unc)
lower_ind = jnp.tril_indices(n_theta)

# ievi RNN and NN
gru_model = sde_cond.RNN(nn_keys[0], n_state, 2*n_meas + 1 + n_theta)
lin_modelr = sde_cond.NN(nn_keys[1], n_state)

# training for full VI
@eqx.filter_value_and_grad
def loss_fn(params, model, key,  y_meas, dt_obs, n_res):
    n_sim = 1
    keys = jax.random.split(key, num=n_sim)
    def vmap_fun(key):
        theta, t_entropy = model.simulate_theta(key, params)
        xsample, entropy = model.simulate(key, params, y_meas, jnp.log(x_init), theta)
        logdens = logdensity(theta, xsample, y_meas, dt_obs, n_res)
        return -(logdens + entropy + t_entropy)
    loss = jnp.sum(jax.vmap(vmap_fun)(keys))
    return loss/n_sim
   
@eqx.filter_jit
def make_step(params, model, key,  y_meas, dt_obs, n_res, opt_state):
    loss, grads = loss_fn(params, model, key,  y_meas, dt_obs, n_res)
    updates, opt_state = optim.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)
    return loss, params, opt_state

for i in range(NUM_LOOPS):
    y_meas = jnp.array(
        pd.read_csv("data/data_{}.csv".format(i)).iloc[:, :2])

    x_init = jax.vmap(lambda x: jnp.interp(sde_times, obs_times, x), in_axes=[1])(y_meas).T
    ievi_model = sde_cond.SmoothModel(n_state, obs_times, sde_times)
    ievi_params = {
        "gru": gru_model,
        "nn": lin_modelr,
        "theta_mu": theta_unc,
        "theta_chol": jnp.eye(n_theta)[lower_ind]*-2
    }

    learning_rate = 0.001
    # optim = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.adam(learning_rate)
    # )
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(ievi_params, eqx.is_array))

    min_loss = jnp.inf
    last_save = 0
    iter = 0
    key = run_key
    while True:
        key, subkey = jax.random.split(key)
        loss, ievi_params, opt_state = make_step(ievi_params, ievi_model, subkey, y_meas, dt_obs, n_res, opt_state)
        loss = loss.item()
        # if iter % 100 == 0:
        #     print(f"step={iter}, loss={loss}")
        #     theta_chol = jnp.zeros((n_theta, n_theta))  
        #     theta_chol = theta_chol.at[lower_ind].set(ievi_params["theta_chol"])
        #     print("theta_mu: ", jnp.exp(ievi_params["theta_mu"]))
        #     print("theta_std: ", jax.nn.softplus(jnp.diag(theta_chol)))

        if jnp.isnan(loss) or loss < 0:
            ievi_params = ievi_params_best
            opt_state = optim.init(eqx.filter(ievi_params, eqx.is_array))
        elif loss < min_loss:
            min_loss = loss
            last_save = iter
            ievi_params_best = ievi_params
        if (iter - last_save > 1000 and loss < 500) or iter >10000:
            print(f"step={iter}, loss={loss}")
            print("theta_mu: ", jnp.exp(ievi_params["theta_mu"]))
            break
        iter += 1

    
    pred_mu = ievi_params_best["theta_mu"]
    theta_chol = jnp.zeros((n_theta, n_theta))  
    theta_chol = theta_chol.at[lower_ind].set(ievi_params_best["theta_chol"])
    pred_se = jax.nn.softplus(jnp.diag(theta_chol))

    df = pd.DataFrame({
        'run_number':i,
        'theta':theta_names, 'mu':pred_mu, 'se':pred_se})
    if os.path.isfile('results/ievi_loglotvol.csv'):
        df.to_csv('results/ievi_loglotvol.csv', mode='a', header=False, index=False) 
    else:
        df.to_csv('results/ievi_loglotvol.csv', index=False)
    
    print("done run {} for ievi".format(i))

# %%


## Ryder -----------------------------------------------------------------------------------
run_key, nn_key = jax.random.split(ryder_key)
n_theta = len(theta_unc)
lower_ind = jnp.tril_indices(n_theta)
lin_model = mr.RyderNN(nn_key, n_state, n_state + n_meas + 2 + n_theta)
ryder_model = mr.RyderModel(n_state, obs_times, sde_times, jnp.eye(n_state), restrict=True)


def reglogdensity(utheta, x_state, y_meas, dt_obs, n_res):
    r"""
    Computes the joint density `p(x,y|theta)`.
    """
    dt_res = dt_obs/n_res
    theta = jnp.exp(utheta)
    # compute latent lpdf
    def state_lpdf(x_curr, x_prev, theta, dt):
        return jsp.stats.multivariate_normal.logpdf(
            x=x_curr,
            mean=x_prev + reglv_model.drift(x_prev, theta) * dt,
            cov=reglv_model.diff(x_prev, theta) * dt
        )
    state_lp = jnp.sum(jax.vmap(lambda xc, xp: 
        state_lpdf(xc, xp, theta, dt_res))(x_state[1:], x_state[:-1]))


    # compute observation lpdf
    def meas_lpdf(y_curr, x_curr):
        return jsp.stats.norm.logpdf(
            x=y_curr,
            loc=x_curr,
            scale=jnp.array([tau_h, tau_l])
        )
    meas_lp = jnp.sum(jax.vmap(meas_lpdf)(y_meas, x_state[::n_res]))
    return state_lp + meas_lp

# training for full VI
@eqx.filter_value_and_grad
def loss_fn(params, model, key, y_meas, dt_obs, n_res):
    n_sim = 1
    keys = jax.random.split(key, num=n_sim)
    def vmap_fun(key):
        theta, t_entropy = model.simulate_theta(key, params)
        xsample, entropy = model.simulate(key, params, y_meas, x0, theta)
        logdens = reglogdensity(theta, xsample, y_meas, dt_obs, n_res)
        return -(logdens + entropy + t_entropy)
    loss = jnp.sum(jax.vmap(vmap_fun)(keys))
    return loss/n_sim
   

@eqx.filter_jit
def make_step(params, model, key,  y_meas, dt_obs, n_res, opt_state):
    loss, grads = loss_fn(params, model, key,  y_meas, dt_obs, n_res)
    updates, opt_state = optim.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)
    return loss, params, opt_state


for i in range(NUM_LOOPS):
    y_meas = jnp.array(
        pd.read_csv("data/data_{}.csv".format(i)).iloc[:, :2])

    ryder_params = {
        "nn": lin_model,
        "theta_mu": theta_unc,
        "theta_chol": jnp.eye(n_theta)[lower_ind]*-2
    }
    
    learning_rate = 0.001
    # optim = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.adam(learning_rate)
    # )
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(ryder_params, eqx.is_array))

    min_loss = jnp.inf
    last_save = 0
    iter = 0
    key = run_key
    while True:
        key, subkey = jax.random.split(key)
        loss, ryder_params, opt_state = make_step(ryder_params, ryder_model, subkey, y_meas, dt_obs, n_res, opt_state)
        loss = loss.item()
        # if iter % 100 == 0:
        #     print(f"step={iter}, loss={loss}")
        #     theta_chol = jnp.zeros((n_theta, n_theta))  
        #     theta_chol = theta_chol.at[lower_ind].set(ryder_params["theta_chol"])
        #     print("theta_mu: ", jnp.exp(ryder_params["theta_mu"]))
        #     print("theta_std: ", jax.nn.softplus(jnp.diag(theta_chol)))

        if jnp.isnan(loss) or loss < 0:
            ryder_params = ryder_params_best
            opt_state = optim.init(eqx.filter(ryder_params, eqx.is_array))
            # print("NAN LOSS")
            
        elif loss < min_loss:
            min_loss = loss
            last_save = iter
            ryder_params_best = ryder_params
        if (iter - last_save > 1000 and loss < 500 and iter > 5000)  or iter >10000:
            print(f"step={iter}, loss={loss}")
            print("theta_mu: ", jnp.exp(ryder_params["theta_mu"]))
            break
        iter += 1

    pred_mu = ryder_params_best["theta_mu"]
    theta_chol = jnp.zeros((n_theta, n_theta))  
    theta_chol = theta_chol.at[lower_ind].set(ryder_params_best["theta_chol"])
    pred_se = jax.nn.softplus(jnp.diag(theta_chol))

    df = pd.DataFrame({
        'run_number':i,
        'theta':theta_names, 'mu':pred_mu, 'se':pred_se})
    if os.path.isfile('results/ryder_loglotvol.csv'):
        df.to_csv('results/ryder_loglotvol.csv', mode='a', header=False, index=False) 
    else:
        df.to_csv('results/ryder_loglotvol.csv', index=False)
        
    print("done run {} for Ryder".format(i))




