# Optimal transitions between non-equilibirum steady states
### Samuel Monter, Sarah A. M. Loos, Clemenes Bechinger

This repository ancompagnies the publication with the mentioned title and is supposed to give the reader acces to the tools used in the second part of the paper. Namely numerical optimization routines using automatic differentiation with `jax` to solve problems from the realm of stochastic thermodynamics.
In order to use the repo, we recommend the setting up a a virtual environment incorporating all packages used by the authors. The procedure for this is descirbed in JAXtutorial.ipynb.

The repo has the following content:
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

We recommend using the optNESStransANN.py to reproduce any results for the paper. It is computationally more efficient. The parametrization by piecewise linear function runs into problems for large protocol times. If other problems shall be tacled using similar routines using the the picewise linear parametrization might by adavantageous as it incorporates discontinuties and start and end of the protocol more easily.
