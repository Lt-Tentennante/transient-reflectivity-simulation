# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import os
file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)
import pymatgen.core as pmg
import xarray as xr
import xraydb

from .constants import hc_nm_eV, re_nm

# MATERIAL - FUNCTIONS
def properties_from_cif(cif_file, egrid, is_cif_print=False):
    # Load structure from CIF file
    structure_from_cif = pmg.Structure.from_file(cif_file)
    if is_cif_print:
        print(structure_from_cif)
        
    # Volume
    cell_volume = structure_from_cif.lattice.volume
    cell_volume_nm3 = cell_volume * 1.e-3  # convert from Å³ to nm³
    
    # Composition and number of elements
    composition = structure_from_cif.composition
    elements = [el.symbol for el in composition]
    
    # Precompute the wavelength in nm for the entire energy grid
    wavelength_nm_grid = hc_nm_eV / egrid  # Convert energy grid to wavelengths in nm
    
    # Create an empty xarray Dataset to store the values for all elements and energies
    ds = xr.Dataset(
        {
            "f1": (["element", "energy"], np.zeros((len(elements), len(egrid)))),
            "f2": (["element", "energy"], np.zeros((len(elements), len(egrid)))),
            "sigma": (["element", "energy"], np.zeros((len(elements), len(egrid)))),
        },
        coords={
            "element": elements,
            "energy": egrid,
        }
    )
    
    # Add a variable to store the absorption coefficient 'mu' over the energy grid
    ds["mu"] = (["energy"], np.zeros(len(egrid)))
    
    # Iterate over elements and compute f1, f2 for the entire energy grid
    for i, el in enumerate(composition):
        count = composition[el]
        f0 = xraydb.f0(el.symbol, [0])
        # Retrieve f1 and f2 for the entire energy grid
        f1_array = xraydb.f1_chantler(el.symbol, egrid) + f0
        f2_array = xraydb.f2_chantler(el.symbol, egrid)
        
        # Calculate sigma for the entire energy grid
        sigma_array = 2 * re_nm * wavelength_nm_grid * f2_array  # Cross-section in nm²
        
        # Store values in the Dataset for this element
        ds["f1"].loc[dict(element=el.symbol)] = f1_array
        ds["f2"].loc[dict(element=el.symbol)] = f2_array
        ds["sigma"].loc[dict(element=el.symbol)] = sigma_array
        
        # Update mu (absorption coefficient) for the entire energy grid
        ds["mu"] += sigma_array * count / cell_volume_nm3

    return ds

def plot_properties(ds):
    """
    General function to plot f1, f2, and 1/mu from any xarray.Dataset.
    
    Parameters:
    ds: xarray.Dataset
        Dataset containing element-specific f1, f2, and mu values as a function of energy.
    """
    elements = ds['element'].values
    energy_grid = ds['energy'].values
    
    # Create a figure with two subplots in a row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot f1 and f2 for each element
    for element in elements:
        ax1.plot(energy_grid, ds["f1"].sel(element=element), label=f'{element} f1')
        ax1.plot(energy_grid, ds["f2"].sel(element=element), '--', label=f'{element} f2')
    
    # Customize the first plot (f1 and f2)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('f1 and f2')
    ax1.set_title('f1 and f2 for each element')
    ax1.legend()
    ax1.grid(True)

    # Plot 1/mu if it exists in the dataset
    mu = ds["mu"].values
    ax2.plot(energy_grid, 1/mu, label='1/mu')

    # Customize the second plot (1/mu)
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('1/mu (nm)')
    ax2.set_title('1/mu vs Energy')
    ax2.set_ylim(0, 1.1 * max(1/mu))
    ax2.grid(True)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

# MATERIAL - CLASS
class Material:
    def __init__(self, cif_filename):
        self.cif_file = directory_path + f'/data_dependencies/cif_files/{cif_filename}'
        self.set_egrids()
        
    def set_egrids(self):
        self.egrids = {'obs': np.linspace(925, 941, num=500),
                       'ext': np.linspace(880, 980, num=1000),
                       'large': np.linspace(925-500, 925+500, num=10000)}

    def set_properties(self, which):
        """
        Reads the properties from the .cif file. The properties are computed for the energy grid specified by 'which'.

        Parameters:
        - which (str): One of ['obs', 'ext', 'large'].
        """
        if which in self.egrids:
            self.properties = properties_from_cif(self.cif_file, self.egrids[which])
        else:
            raise ValueError(f"Invalid choice of egrid: {which}. Only available choices are 'obs', 'ext', or 'large'.")

        return self.properties      