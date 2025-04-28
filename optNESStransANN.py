# This script optimize the vi => vf in finite time control problem
# the optimal control protocol is approximated using an artificial neural network (ANN)
# the optimization can be performed using differen physical models for the system under control:
# an overdamped langevin equation or an overdamped maxwell modell with 1 or 2 addition degree of freedom


import jax.numpy as jnp
from jax import config

import optax                # optimization package for JAX

from flax import nnx

import numpy as np
import matplotlib.pyplot as plt

from util.simulation import simulation2
from util.simulation import harmonic
from util.thermodynamics import WofXandL
from util.parametrization import parametrization
from util.makeANN import ANN_1_n_1

from datetime import datetime
import subprocess
import json

# set precision to float64 to prevent numerical erros present in default float32 precision
config.update("jax_enable_x64",True)


def beep():
    subprocess.run("powershell.exe '[console]::beep(440, 500)'   ",shell=True)

# initialize all necessary objects and functions for simulation and work calculation
def init_env(params):
    protocol_func =  parametrization(params).make_protocol_ANN # select a protocol function, this script is optimized for ANN usage
    pot = harmonic()                            # potential function
    sim = simulation2(params,pot.potential,3)   # simulation function including update and loop
    Wfunc = WofXandL(params,pot.potential)      # object with method to compute work
    return protocol_func, pot, sim, Wfunc

# initalize optimizer
def init_opt(learning_schedule,controler):
    optimizer = nnx.Optimizer(controler,
                              optax.adam(learning_schedule,b1=0.95,b2=0.99))
    return optimizer

# final function that combines: protocol generation,
# simulation and work calculation
@nnx.jit
def loss_func(controler):
    protocol = protocol_func(controler)
    traj = sim.run(params["x0"],protocol)
    W = Wfunc.calculate2_perfectjump(traj,protocol)
    return W

# one optimizatin step
@nnx.jit
def opt_step(controler,optimizer):
    loss, grads = nnx.value_and_grad(loss_func)(controler)
    optimizer = optimizer.update(grads)             # apply update
    return loss

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
        "Nprot": 60000,                 # sim for protocol, ts = Nprot * dt
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

# set optimization parameters
num_opt_it = 1000                # number of optimization steps
init_learning_rate = 1*1e-2     # initial learnir rate
schedule_boundary_frct  = [0.9, 0.99]   # relative time to change learning rate
learning_rate_ratio = [1*1e-1, 1e-1]    # ratio of learning rate after step

# create learning shedule
learning_schedule = optax.piecewise_constant_schedule(
    init_value = init_learning_rate,
    boundaries_and_scales = {int(schedule_boundary_frct[i]*num_opt_it):
                             learning_rate_ratio[i] for i in range(len(schedule_boundary_frct))})

# create ANN
controler = ANN_1_n_1(16,.1,1) # n hidden, scale, seed

# initialize environment
protocol_func, pot, sim, Wfunc = init_env(params)

# init optimizer with fixed state
optimizer = init_opt(learning_schedule,controler)

print("preparation done: ", datetime.now())

opt_start_time = datetime.now()

# for collecting results
losses_fin = []

for i in range(num_opt_it):
    losses_fin.append(opt_step(controler,optimizer))

losses_fin = jnp.array(losses_fin)

print("opt took :", datetime.now()-opt_start_time)
print("W_opt = ", losses_fin[-1])
print("startplotting", datetime.now())


## plot results
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# loss as function of optimization
ax1.semilogx(losses_fin)#-losses_fin[best_ind,-1])
ax1.set_xlabel("optimization steps")
ax1.set_ylabel(r'$\langle W_{ex} \rangle$')
ax1.vlines(num_opt_it*jnp.array(schedule_boundary_frct),losses_fin[-1],
           jnp.max(jnp.array(losses_fin)),color="black",linestyle="--")


# best result: protocol, trajs
protocol = protocol_func(controler)
traj = sim.run(params["x0"],protocol)

Nstep = params["Npost"] + params["Npre"] + params["Nprot"]
t = jnp.linspace(0,Nstep-1,Nstep) * params["dt"]


ax2.plot(t[0:params["Nprot"]+params["Npre"]+10],
         protocol[0:params["Nprot"]+params["Npre"]+10],label=r'$\lambda$')
ax2.plot(t[0:params["Nprot"]+params["Npre"]+10],
         traj[0:params["Nprot"]+params["Npre"]+10,0],linestyle = "--",label=r'$x_T$')
ax2.plot(t[0:params["Nprot"]+params["Npre"]+10],
         traj[0:params["Nprot"]+params["Npre"]+10,1],linestyle = "--",label=r'$x_B1$')
ax2.plot(t[0:params["Nprot"]+params["Npre"]+10],
         traj[0:params["Nprot"]+params["Npre"]+10,2],linestyle = "--",label=r'$x_B2$')
ax2.legend
ax2.set_xlabel("time")
ax2.legend()
