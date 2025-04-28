import jax.numpy as jnp
from jax import jit         # just in time compliation for faster execution
from jax import vmap        # vectorizing functions to accelerated computation with arrrays

from flax import nnx        # neural network package


class parametrization:
    def __init__(self,params_in):
        global params
        params = params_in
        '''
        # this is bad practive ... but lets see
        for key, value in params.items():
            #print(key)
            #print(value)
            globals()[key] = value
        '''
    # together with jittesfuc.clear_cache() this allows setup with new params
    @staticmethod
    def setparams(params_in):
        global params
        params = params_in
        return params

    # simplext protocol, trap velocity = 0, just to jumps at t= 0, t= ts
    @staticmethod
    @jit
    def make_protocol_jj(js):
        li = jnp.cumsum(jnp.ones(params["Npre"])*params["vi"]*params["dt"])
        lp = jnp.cumsum(jnp.zeros(params["Nprot"])) + li[-1] + js[0]
        lf = jnp.cumsum(jnp.ones(params["Npost"])*params["vf"]*params["dt"]) + lp[-1] + js[1]
        return jnp.concat([li,lp,lf])


    # picewise linear
    @staticmethod
    @jit # takes 100x longer without jit, no difference if in class
    def make_protocol_piecelin(ls):
        n = jnp.linspace(1,params["Nprot"],params["Nprot"])         # time steps in protocol, choose these inputs to cover both jumps
        dn = params["Nprot"]/(len(ls)-2)       # time steps per segment

        # use len(ll)-2 because: n-1 segments, -1 because last lambda is just jump
        # len(funlist) has to be len(condlist) + 1

        condlist = [jnp.logical_and(i*dn <= n, n < (i+1)*dn)
                    for i in range(len(ls)-2)]  # list of conditions segmenting the protocol

        funclist = [lambda n, i=i: (ls[i+1]-ls[i])/dn*n + (i+1)*ls[i] - i*ls[i+1]
                    for i in range(len(ls)-2)] # linear function for each segment

        funclist.append(ls[-1])                 # ppend jump
        lp = jnp.piecewise(n, condlist, funclist) # calculate protocol

        #  stitch together whole lambda
        li = jnp.cumsum(jnp.ones(params["Npre"])*params["vi"]*params["dt"])
        lp = lp + li[-1]
        lf = jnp.cumsum(jnp.ones(params["Npost"])*params["vf"]*params["dt"]) + lp[-1]
        return jnp.concat([li,lp,lf])


    # polynomial
    @staticmethod
    @jit
    def make_protocol_poly(coefs):
        n = jnp.linspace(0,1,params["Nprot"]) # time steps in protocol, choose these inputs to cover both jumps
        n = n - 0.5 #jnp.floor(params["Nprot"]/2)      # put 0 in midle of protocol, in expectation of some kind of symmetry

        lp = jnp.polyval(coefs[:-1],n)   # last value of coefs is not used for polynomial but for last jump
        lp = lp.at[-1].add(coefs[-1])   # add final jump in protocol
        lp = lp

        #  stitch together whole lambda
        li = jnp.cumsum(jnp.ones(params["Npre"])*params["vi"]*params["dt"])
        lp = lp + li[-1]
        lf = jnp.cumsum(jnp.ones(params["Npost"])*params["vf"]*params["dt"]) + lp[-1]
        return jnp.concat([li,lp,lf])


    # neural, ANN are created in util.makeANN.py
    @staticmethod
    @nnx.jit # need to use this version of jit
    def make_protocol_ANN(controler):
        n = jnp.linspace(0,1,params["Nprot"]).reshape(-1,1) # normalize ANN input between 0 and 1, need to reshape for vmap to work

        lp = vmap(controler)(n) # predict protocol
        lp = lp * 1e-6

        #  stitch together whole lambda
        li = jnp.cumsum(jnp.ones(params["Npre"])*params["vi"]*params["dt"])
        lp = lp + li[-1]                                # here a discontinuity can easily apear if lp[0] != 0
        lf = jnp.cumsum(jnp.ones(params["Npost"])*params["vf"]*params["dt"]) + lp[-1] # this make it hard for jump to apear
        return jnp.concat([li,jnp.squeeze(lp),lf])
