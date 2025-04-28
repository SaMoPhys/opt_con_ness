# This script optimize the vi => vf in finite time control problem
# the optimal control protocol is approximated using a piecewise linear function
# the optimization can be performed using differen physical models for the system under control:
# an overdamped langevin equation or an overdamped maxwell modell with 1 or 2 addition degree of freedom

import jax.numpy as jnp
from jax import jit         # just in time compliation for faster execution
from jax import value_and_grad
from jax.lax import scan    # JAX optimized version of a FOR loop
from jax import random
from jax import config

import optax                # optimization package for JAX

import numpy as np

import matplotlib.pyplot as plt

from util.simulation import simulation2
from util.simulation import harmonic
from util.thermodynamics import WofXandL
from util.parametrization import parametrization

from datetime import datetime
import subprocess
import json

# set precision to float64 to prevent numerical erros present in default float32 precision
config.update("jax_enable_x64",True)

def beep():
    subprocess.run("powershell.exe '[console]::beep(440, 500)'   ",shell=True)

# initialize all necessary objects and functions for simulation and work calculation
def init_env(params):
    protocol_func =  parametrization(params).make_protocol_piecelin # select a protocol function - this script is optimized for piecelin, should also run with polynomial
    pot = harmonic()                            # potential function
    sim = simulation2(params,pot.potential,3)   # simulation function including update and loop, 2nd argument selects simulation model
    Wfunc = WofXandL(params,pot.potential)      # object with method to compute work
    return protocol_func, pot, sim, Wfunc

# initalize optimizer
def init_opt(start_learning_rate,opt_params):
    optimizer = optax.adam(start_learning_rate,b1=0.8,b2=0.99,eps=1e-8)
    opt_state = optimizer.init(opt_params)
    return optimizer, opt_state

# final function that combines: protocol generation,
# simulation and work calculation
@jit
def loss_func(input):
    protocol = protocol_func(input*1e-6)
    traj = sim.run(params["x0"],protocol)
    W = Wfunc.calculate2(traj,protocol)
    return W

# one optimizatin step
@jit
def opt_step(carry,_):
  opt_params, opt_state = carry                             # unpack carry
  loss, grads = value_and_grad(loss_func)(opt_params)       # calculate loss and gradient
  updates, opt_state = optimizer.update(grads, opt_state)   # calculate update
  opt_params = optax.apply_updates(opt_params, updates)     # apply update
  return (opt_params, opt_state), loss

starttime = datetime.now()

# define parameters whcih define model parameters and boundary condition vor optimization
params = {"k": 4.47661726865097e-06,    # trap stiffness, N/m
        "gT":0.18551532*1e-6,           # fricition tracer, Ns/m
        "gB1":0.31927638*1e-6,          # friciton 1st bath particlem Ns/m
        "kB1":0.57299359*1e-6,          # stiffnes 1st bath spring, N/m
        "gB2":0.17884259*1e-6,          # friticion 2nd bath particle, Ns/m
        "kB2":0.03010254*1e-6,          # stiffnes 2nd bath spring, N/m
        "dt": 1e-4,                     # integration step, s
        "Npre": 100,                    # sim steps before protocol
        "Nprot": 600,                   # sim for protocol, ts = Nprot * dt
        "Npost": 100,                   # sim steps after protocol
        "vi": 1.*1e-6,                  # inital trap speed, m/s
        "vf": 0.0*1e-6,                 # final trap speed, m/s
        "x0":None}                      # particle starting positions, calculate later



# calculate steady state positions of tracer and 2 bath particles
xT = -params["vi"]*(params["gT"]+params["gB1"]+params["gB2"])/params["k"] # <0
dx1 = -params["vi"]*params["gB1"]/params["kB1"]
dx2 = -params["vi"]*params["gB2"]/params["kB2"]
params["x0"] = jnp.array([xT,xT+dx1,xT+dx2])

'''
# calculate steady state positions of tracer and one bath particle
xT = -params["vi"]*(params["gT"]+params["gB"])/params["k"] # <0
dx = -params["vi"]*params["gB"]/params["kB"]
params["x0"] = jnp.array([xT,xT+dx])
# calculate steady state position of tracer without any bath particles
params["x0" ] = -jnp.array([params["vi"]*params["gT"]/params["k"]])
'''

# initialize environment
protocol_func, pot, sim, Wfunc = init_env(params)

# set optimization parameters
num_opt_param = 30              # number of parameters to optimizes i.e. number of nodes in lin.param.fct.
num_batch = 3                   # number of optmization runs to execute
num_opt_it = 100                # number of optimization steps per run
init_learning_rate = 0.2        # initial learnir rate
schedule_boundary_frct  = [0.3,0.6]   # relative time to change learning rate
learning_rate_ratio = [1*1e-1,1*1e-1] # ratio of learning rate after step

# create learning shedule
learning_schedule = optax.piecewise_constant_schedule(
    init_value = init_learning_rate,
    boundaries_and_scales = {int(schedule_boundary_frct[i]*num_opt_it):
                             learning_rate_ratio[i] for i in range(len(schedule_boundary_frct))})

# init optimizer with fixed state
optimizer, opt_state = init_opt(learning_schedule,jnp.zeros(num_opt_param))

# generate random start parameters for whole batch
key = random.key(0)
opt_params_batch = jnp.ones((num_batch,num_opt_param)) * random.uniform(key,(num_batch,1),minval=-2,maxval=2)

# prepare result collection
opt_params_fin = np.empty([num_batch,num_opt_param])    # final optimized parameters
losses_fin = np.empty([num_batch,num_opt_it])           # loss trajectory

print("preparation done: ", datetime.now())

for i in range(num_batch):
    scan_start_time = datetime.now()
    print("Performing scan number ", i," :", scan_start_time)

    # init optimizer with new opt params
    opt_state = optimizer.init(opt_params_batch[i,:])

    # run optimization
    #  for this a normla for loop would be just as fast...
    (opt_params_out, opt_state), losses = scan(opt_step,
                                               (opt_params_batch[i,:], opt_state),None,
                                               length=num_opt_it)

    # collect results
    opt_params_fin[i,:] = opt_params_out
    losses_fin[i,:] = losses
    print("scan took :", datetime.now()-scan_start_time," s")

print("startplotting", datetime.now())

# find best opt run
best_ind = np.argmin(losses_fin[:,-1])
print("W_opt = ", losses_fin[best_ind,-1],"kT")

## plot results
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

# loss as function of optimization
ax1.semilogx(losses_fin.transpose())
ax1.set_xlabel("optimization steps")
ax1.set_ylabel(r'$\langle W_{ex} \rangle$')
ax1.vlines(num_opt_it*schedule_boundary_frct,losses_fin[best_ind,-1],
           jnp.max(losses_fin[:,0]),color="black",linestyle="--")

# parameters of evry batch
for i in range(num_batch):
    ax2.scatter(range(num_opt_param),opt_params_fin[i,:],marker='x')

ax2.set_xlabel("parameters")
ax2.set_ylabel('value')

# best result: protocol, trajs
protocol = protocol_func(opt_params_fin[best_ind,:]*1e-6)
traj = sim.run(params["x0"],protocol)

Nstep = params["Npost"] + params["Npre"] + params["Nprot"]
t = jnp.linspace(0,Nstep-1,Nstep) * params["dt"]


#plot during protocol
ax3.plot(t[0:params["Nprot"]+params["Npre"]+10],
         protocol[0:params["Nprot"]+params["Npre"]+10],label=r'$\lambda$')
ax3.plot(t[0:params["Nprot"]+params["Npre"]+10],
         traj[0:params["Nprot"]+params["Npre"]+10,0],linestyle = "--",label=r'$x_T$')
ax3.plot(t[0:params["Nprot"]+params["Npre"]+10],
         traj[0:params["Nprot"]+params["Npre"]+10,1],linestyle = "--",label=r'$x_B1$')
ax3.plot(t[0:params["Nprot"]+params["Npre"]+10],
         traj[0:params["Nprot"]+params["Npre"]+10,2],linestyle = "--",label=r'$x_B2$')
ax3.legend
ax3.set_xlabel("time")
ax3.legend()
