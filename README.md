# Optimal transitions between non-equilibirum steady states
### Samuel Monter, Sarah A. M. Loos, Clemenes Bechinger

This repository accompanies the publication of the same title and provides access to the numerical tools used in the study. Specifically, it contains the code for the numerical optimization routines based on automatic differentiation with JAX, developed to solve problems of thermodynamically optimal control. To use the repository, it is recommended to set up a virtual environment containing all required packages and using python 3.12.7, as described in notebooks/JAXtutorial.ipynb.

The repository has the following content:
- notebooks
    - JAXtutorial.ipynb: a short intro to `jax`, solving the Schmiedel & Seifert problem
- util: variety of classes used in the main scripts
    - makeANN.py: creating simple artifical neural networks
    - parameterization.py: different ways of parametrizing protocols
    - simulation.py: implementation of langevin dynmics simulations
    - thermodynamics.py: routines to calculate work from simulated trajectories and protocols
- optNESStrans.py: full routine to find the optimal protocol for vi=>vf transition, the protocol is parametrized by a piecwise linear function
- optNESStransANN.py: full routine to find the optimal protocol for vi=>vf transition, the protocol is parametrized by a artifical neural network
- README.md: -
- requirements.txt: needed for setup of virtual enviroment to make the repo run
- setup.py: needed for setup of virtual enviroment to make the repo run

For the optimal control problems considered in this paper, we recommend using optNESStransANN.py, as it is computationally more efficient than optNESStrans.py. The piecewise linear parameterization used in optNESStrans.py can encounter difficulties for large protocol durations. However, for other optimal control problems, the piecewise linear parameterization may be advantageous, as it more easily incorporates discontinuities at the start and end of the protocol.
