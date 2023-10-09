# KineticAssembly_AD
Python code for numerical simulations of self-assemly with optimization by automatic differentiation

## Installation ##

The easiest way to install the simulator is to clone this repo and then build an environment containing all dependencies using the provided `base_requirements.txt` file. In order to do this you will need to have an up to date version of the anaconda package manager (https://www.anaconda.com/products/individual#Downloads). 

- first clone this repository into the desired directory on your system. `git clone git@github.com:mjohn218/multi_assembly.git`
- From the current directory and run `conda create --name <env> --file base_requirements.txt` where `<env>` is the desired name of your new environment. *NOTE: This requirements file only includes dependencies available from conda or pip. For any application involving rosetta, for example estimating free energies from pdb structures, you will need to also install pyrosetta to your environment from http://www.pyrosetta.org/*
- now run `conda activate <env>` in order to use the new environment. 
- you may now use the included modules.

## Documentation ##
Detailed functionality and documentation can be found in the Jupyter Notebooks located in the `docs` directory. 
You can start the jupyter server by activating the conda environment and then running `jupyter notebook`. This should open a browser window
showing the current directory. You can then open the `docs` folder and then any of the notebooks therewithin.

Essentially, the module consists of the following components:

- The `ReactionNetwork` class. Provides methods for generating a full set of possible system states and interactions from a list of pairwise rule and free energies specified in a input `.pwr` file. 
- The `VectorizedReactionNetwork` class. Takes a `ReactionNetwork` as input and converts its networkx explicit graph representation into a pytorch Tensor representatio. Also provides methods for some computations on the system
- The `VecSim` class. A vectorized deterministic simulator runs a rule based simulation on the system specified by a `VectorizedReactionNetwork`
- The `Optimizer` class. Runs a certain number of simulations and adjusts input parameters in order to optimize a metric of complex yield.
- The `EqSolver` class. Takes a `ReactionNetwork` as input and finds the equilibrium solution using a numerical solver. 

- The `EnergyExplorer` class. Somewhat sepeperate. This takes a reaction network and a directory of pdbs for each of the monomer subunits, and generates a list of approximate free energies for each of the pairwise reactions. Requires pyrosetta to be installed.

## Results ##
Similar to the above, some initial results are contained in notebooks located in the `results` folder. 
