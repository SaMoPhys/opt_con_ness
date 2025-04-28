# here we define different approaches to calculate work from particl trajectories and protocols
# all use the stratonivich formalism to numerically integrate the expression from stoch. thermodynamics for work
# W = int dV/dl dl/dt dl
# some approaches use some analytical expressions to calculate parts of the total work
# since we are occupied with linear system noise is ommitted in all cases

import jax.numpy as jnp
from jax import jit         # just in time compliation for faster execution
from jax import grad        # auto grad - thats what we are here for
from jax import vmap        # vectorizing functions to accelerated computation with arrrays

class WofXandL:
    def __init__(self,params_in, potential_in):
        global params
        params = params_in

        global potential
        potential = potential_in
    # calculate work from trajectory and protocol using stratanovich convention

    # together with jittesfuc.clear_cache() this allows setup with new params
    @staticmethod
    def setparams(params_in, potential_in):
        global params
        params = params_in

        global potential
        potential = potential_in
        return params, potential

    # calculate excess work fully from simulation
    @staticmethod
    @jit
    def calculate(traj,protocol):
        # vmap vectorizes the computation and therefor accelerats it
        # always take the 0th column of traj (should be tracer in maxwell like models)
        dVdl = vmap(grad(potential,argnums=1),(0,0,None))(traj[:,0],protocol,params["k"])
        dVdlstr = (dVdl[1:] + dVdl[:-1])/2

        dl = jnp.diff(protocol,1,0)

        # -1 index is correct, i checked
        W = jnp.sum(dVdlstr[params["Npre"]-1:] * dl[params["Npre"]-1:])

        # housekeeping work
        # ANALYTICAL DEFINITION
        # due to numeric precision this is not the value in sim => leads to convegence inssues
        # WHK = params["Npost"] * params["dt"] * params["vf"]**2 * (params["gT"] + params["gB"])

        # NUMERICAL DEFINITION
        WHK = dVdlstr[-1] * dl[-1] * params["Npost"]

        return (W - WHK) / ((273.15+25) * 1.38*1e-23) # excess wokr

    # used
    # calculate Wex(0<t<tf) from sim and Wex(tf<t) from analytics
    @staticmethod
    @jit
    def calculate2(traj,protocol):
        # analytical expression for Wex for a constant driving starting from any inital condition of particle positions
        # for 2 bath particle model
        def W_ex_after_calc(gamma, gamma1, gamma2, v, kappa1, kappa2, k, x0, y01, y02):
            numerator = (
                        gamma**2 * gamma1**2 * gamma2**2 * v *
                        ((kappa1**2 * kappa2**2 * v) / (gamma**2 * gamma1**2) +
                        (kappa2**2 * ((kappa1**2 * v) / gamma**2 +
                                        (kappa1**2 * (v + (k * x0) / gamma)) / gamma1**2 +
                                        (kappa1 * ((k * v) / gamma + (2 * kappa1 * v) / gamma1 +
                                                (k * kappa1 * y01) / (gamma * gamma1))) / gamma)) / gamma2**2 +
                        (kappa1 * kappa2 * ((2 * kappa1 * kappa2 * v) / (gamma * gamma2) +
                                            (kappa1 * ((k * v) / gamma + (2 * kappa2 * v) / gamma2 +
                                                        (k * kappa2 * y02) / (gamma * gamma2))) / gamma1)) / (gamma * gamma1)
                        )
                        )

            denominator = k**2 * kappa1**2 * kappa2**2

            return -k * numerator / denominator

        # vmap vectorizes the computation and therefor accelerats it
        # always take the 0th column of traj (should be tracer in maxwell like models)
        dVdl = vmap(grad(potential,argnums=1),(0,0,None))(traj[:,0],protocol,params["k"])
        dVdlstr = (dVdl[1:] + dVdl[:-1])/2

        dl = jnp.diff(protocol,1,0)

        P = dVdlstr[:] * dl[:]
        # I checked the time intervals in anna_Wex_after_tf.ipynb
        Wprot = jnp.sum(P[params["Npre"]-1:params["Npre"]+params["Nprot"]-1])

        # particle positions after protocol finished
        xtf = traj[params["Npre"]+params["Nprot"]-1,:] - protocol[params["Npre"]+params["Nprot"]-1]

        # exces work after ts
        Wafter = W_ex_after_calc(params["gT"], params["gB1"], params["gB2"],
                                 params["vf"], params["kB1"], params["kB2"],
                                 params["k"], xtf[0],xtf[1],xtf[2])

        return (Wprot + Wafter) / ((273.15+25) * 1.38*1e-23)

    # used
    # calculate Wex(0<t<tf) from sim and Wex(tf<t) from analytics
    # also clalculate perfect jump distance at tf - which means that the protocol does not need to contain the final jump
    # this is very usefull since ANN protocol parametrization does not give jump at end of protocol
    @staticmethod
    @jit
    def calculate2_perfectjump(traj,protocol):
        # analytical expression for Wex for a constant driving starting from any inital condition of particle positions
        # for 2 bath particle model
        def W_ex_after_calc(gamma, gamma1, gamma2, v, kappa1, kappa2, k, x0, y01, y02):
            numerator = (
                        gamma**2 * gamma1**2 * gamma2**2 * v *
                        ((kappa1**2 * kappa2**2 * v) / (gamma**2 * gamma1**2) +
                        (kappa2**2 * ((kappa1**2 * v) / gamma**2 +
                                        (kappa1**2 * (v + (k * x0) / gamma)) / gamma1**2 +
                                        (kappa1 * ((k * v) / gamma + (2 * kappa1 * v) / gamma1 +
                                                (k * kappa1 * y01) / (gamma * gamma1))) / gamma)) / gamma2**2 +
                        (kappa1 * kappa2 * ((2 * kappa1 * kappa2 * v) / (gamma * gamma2) +
                                            (kappa1 * ((k * v) / gamma + (2 * kappa2 * v) / gamma2 +
                                                        (k * kappa2 * y02) / (gamma * gamma2))) / gamma1)) / (gamma * gamma1)
                        )
                        )

            denominator = k**2 * kappa1**2 * kappa2**2

            return -k * numerator / denominator

        # vmap vectorizes the computation and therefor accelerats it
        # always take the 0th column of traj (should be tracer in maxwell like models)
        dVdl = vmap(grad(potential,argnums=1),(0,0,None))(traj[:,0],protocol,params["k"])
        dVdlstr = (dVdl[1:] + dVdl[:-1])/2

        dl = jnp.diff(protocol,1,0)

        P = dVdlstr[:] * dl[:]
        # I checked the time intervals in anna_Wex_after_tf.ipynb
        Wprot = jnp.sum(P[params["Npre"]-1:params["Npre"]+params["Nprot"]-1])

        #  calculation of perfect jump: We know that after the protocol x - l = -x_NESS
        # this means: -x_NESS = x(tf) - (l(tf) + dl)
        #               dl = x_NESS + x(tf) - l(tf)

        # relative position after protocol
        xtf = traj[params["Npre"]+params["Nprot"]-1,:] - protocol[params["Npre"]+params["Nprot"]-1]
        # relative position in final NESS
        xN = - params["vf"] *(params["gT"] + params["gB1"] + params["gB2"]) / params["k"]
        dl = xtf[0] + xN #  the particle must end in the negative relative NESS position

        dV = potential(xtf[0], dl,params["k"]) - potential(xtf[0], 0,params["k"])

        Wafter = W_ex_after_calc(params["gT"], params["gB1"], params["gB2"],
                                 params["vf"], params["kB1"], params["kB2"],
                                 params["k"], xtf[0] - dl,xtf[1] - dl,xtf[2] - dl)

        return (Wprot + dV + Wafter) / ((273.15+25) * 1.38*1e-23)
