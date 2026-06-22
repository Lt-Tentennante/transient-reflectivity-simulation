# Simulation of the transient reflectivity of a nanostructured multilayer
To run the notebooks contained in this folder you need a custom environment. Follow the instructions in `./env_creation/README.md` to setup the custom environment.

## 2level_fit.ipynb
This notebook is used to fit the X-Ray Reflectivity (XRR) curve to the experimental data. The notebook relies on the XRR module to simulate a reflectivity curve where the scattering factor of Cu atoms contains a resonant contribution. The parameters of this resonance are to be fitted against experimental data.

### Paper figure
The last section of this notebook is used to produce the figure that appears in the paper. The experimental data is loaded from memory, the fit result are simply manually saved in arrays and used to simulate the curve.

## self_consistent_simulation.ipynb
Uses XRR and TASsello to simulate the reflectivity curves to later be compared to the experimental measurements. The results can (and should be) stored for later use.

## plot_simulation_results.ipynb
This notebook uses the results of `self_consistent_simulation.ipynb` to plot the fluence series of the simulation. It uses the results stored in `./sources/self_consistent_results/`. The final cell of the notebook produces the figure in the paper.