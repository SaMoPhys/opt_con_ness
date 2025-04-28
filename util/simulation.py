import jax.numpy as jnp
from jax import jit         # just in time compliation for faster execution
from jax import grad        # auto grad - thats what we are here for
from jax.lax import scan    # JAX optimized version of a FOR loop

## markovian bath
# update step for langevin simulation at 0K (no noise)

class simulation:
    # slower - might be removed at some point
    def __init__(self, params_in, potential_in, model):
        global params
        params = params_in

        global potential
        potential = potential_in

        self.model = model

    @staticmethod
    def setparams(params_in, potential_in):
        #global params
        params = params_in

        #global potential
        potential = potential_in

    # update function for overdamped langevin equation
    @staticmethod
    @jit
    def update_langevin(x,l):
        dx = - grad(potential,argnums=0)(x[0],l,params["k"])/ params["gT"] * params["dt"]
        x = x.at[0].add(dx)
        # we give this weird output due to the special looping function that we use later
        # might no be optimal
        return x, x

    # update function for overdampedmaxwell modell with 1 addition particle
    @staticmethod
    @jit
    # x = [tracer, bath]
    def update_maxwell(x,l):
        ff = params["kB"] * (x[0] - x[1]) # ficticius force
        dxT = (- ff - grad(potential,argnums=0)(x[0],l,params["k"])) / params["gT"] * params["dt"]
        dxB = ff / params["gB"] * params["dt"]
        x = x + jnp.array([dxT, dxB])
        return x, x

    # update function for overdampedmaxwell modell with 2 addition particles
    @staticmethod
    @jit
    # x = [tracer, bath1, bath2]
    def update_maxwell2(x,l):
        ff1 = params["kB1"] * (x[0] - x[1]) # ficticius force
        ff2 = params["kB2"] * (x[0] - x[2]) # ficticius force
        dxT = (- ff1 -ff2 - grad(potential,argnums=0)(x[0],l,params["k"])) / params["gT"] * params["dt"]
        dxB1 = ff1 / params["gB1"] * params["dt"]
        dxB2 = ff2 / params["gB2"] * params["dt"]
        x = x + jnp.array([dxT, dxB1, dxB2])
        return x, x

    # whole simulation
    def run(self,x0,protocol):
        # the scan function is an optimized version of a for loop in JAX
        # input: function executed in loop, starting value, list to loop over
        # output: carry = first output of function, stacked array of second output of fucntion
        print(self.model)
        if self.model == 1:
            _, traj = scan(self.update_langevin,x0,protocol)
        elif self.model == 2:
            _, traj = scan(self.update_maxwell,x0,protocol)
        elif self.model == 3:
            print("da")
            _, traj = scan(self.update_maxwell2,x0,protocol)
        # the 0th column of traj should always be the tracer in maxwellian modells
        return traj


class harmonic:

    @staticmethod
    @jit
    def potential(x,l,k):
        return k/2 * (x-l)**2

# same as above but optimized, update functions are nested with in run function
class simulation2:
    # faster by x10 if run by itseld
    # for gradient based optimization routine irrelevant
    # further optimiized to jit whole simulation and not just update
    def __init__(self, params_in, potential_in, model_in):
        global params
        params = params_in

        global potential
        potential = potential_in

        global model
        model = model_in

    # together with jittesfuc.clear_cache() this allows setup with new params
    @staticmethod
    def setparams(params_in, potential_in):
        global params
        params = params_in

        global potential
        potential = potential_in
        return params, potential


    # whole simulation
    @staticmethod
    @jit
    def run(x0,protocol):

        # update function for overdamped langevin equation
        @jit
        def update_langevin(x,l):
            dx = - grad(potential,argnums=0)(x[0],l,params["k"])/ params["gT"] * params["dt"]
            x = x.at[0].add(dx)
            # we give this weird output due to the special looping function that we use later
            # might no be optimal
            return x, x

        # update function for overdampedmaxwell modell with 2 addition particles
        @jit
        # x = [tracer, bath]
        def update_maxwell(x,l):
            ff = params["kB"] * (x[0] - x[1]) # ficticius force
            dxT = (- ff - grad(potential,argnums=0)(x[0],l,params["k"])) / params["gT"] * params["dt"]
            dxB = ff / params["gB"] * params["dt"]
            x = x + jnp.array([dxT, dxB])
            return x, x

        # update function for overdampedmaxwell modell with 2 addition particles
        @jit
        # x = [tracer, bath1, bath2]
        def update_maxwell2(x,l):
            ff1 = params["kB1"] * (x[0] - x[1]) # ficticius force
            ff2 = params["kB2"] * (x[0] - x[2]) # ficticius force
            dxT = (- ff1 -ff2 - grad(potential,argnums=0)(x[0],l,params["k"])) / params["gT"] * params["dt"]
            dxB1 = ff1 / params["gB1"] * params["dt"]
            dxB2 = ff2 / params["gB2"] * params["dt"]
            x = x + jnp.array([dxT, dxB1, dxB2])
            return x, x

        # the scan function is an optimized version of a for loop in JAX
        # input: function executed in loop, starting value, list to loop over
        # output: carry = first output of function, stacked array of second output of fucntion
        if model == 1:
            _, traj = scan(update_langevin,x0,protocol)
        elif model == 2:
            _, traj = scan(update_maxwell,x0,protocol)
        elif model == 3:
            _, traj = scan(update_maxwell2,x0,protocol)
        # the 0th column of traj should always be the tracer in maxwellian modells
        return traj
