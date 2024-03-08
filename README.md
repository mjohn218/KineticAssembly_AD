# KineticAssembly_AD
Python code for numerical simulations of self-assemly with optimization by automatic differentiation

## Installation ##

The easiest way to install the simulator is to clone this repo and then build an environment containing all dependencies using the provided `base_requirements.txt` file. In order to do this, you will need to have an up to date version of the [Anaconda package manager](https://www.anaconda.com/products/individual#Downloads). 

- First, clone this repository into the desired directory on your system with `git clone https://github.com/mjohn218/KineticAssembly_AD.git`.
- Enter the directory `KineticAssembly_AD` and run `conda create --name <env> --file base_requirements.txt`, where `<env>` is the desired name of your new environment. *NOTE: This requirements file only includes dependencies available from conda or pip. For any application involving Rosetta, e.g. estimating free energies from PDB structures, you will need to also install [PyRosetta](http://www.pyrosetta.org) in the environment.*
- Run `conda activate <env>` to activate the new environment.

## Documentation ##
Detailed functionality and documentation can be found in the Jupyter Notebooks located in the `docs` directory. A UserGuide is provided in the docs folder which desribes the requirements of an input file. Further detailed instructions on creating Reaction Networks, running simulations and optimization can be found within jupyter notebooks in the 'results' folder.
You can start the Jupyter server by activating the conda environment and running the command `jupyter notebook`. This should open a browser window showing the current directory. You can then open the `docs` folder and then any of the notebooks therewithin.

The module consists of the following components:

- The `ReactionNetwork` class provides methods for generating a full set of possible system states and interactions from a list of pairwise rule and free energies specified in a input `.pwr` file. 
- The `VectorizedReactionNetwork` class takes a `ReactionNetwork` object as input and converts its NetworkX explicit graph representation into a PyTorch tensor representation. Also provides methods for some computations on the system.
- The `VecSim` class, in which a vectorized deterministic simulator runs a rule based simulation on the system specified by a `VectorizedReactionNetwork`.
- The `Optimizer` class runs a certain number of simulations and adjusts input parameters in order to optimize a metric of complex yield.
- The `EqSolver` class takes a `ReactionNetwork` object as input and finds the equilibrium solution using a numerical solver. 
- The `EnergyExplorer` class is somewhat separate. It takes as input a `ReactionNetwork` object and and a directory of .pdb files, one for each subunit monomer, and generates a list of approximate free energies for each of the pairwise reactions. Requires PyRosetta.

## Results ##
The 'results' folder contains jupyter notebooks for different types of optimization protocols as outlined in the paper. 
