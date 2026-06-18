import numpy as np
from scipy.interpolate import interp1d

from .constants import hc_nm_eV, re_nm, Cu_n_nm3
from .MaterialCreator import Material

Cu = Material('ICSD_CollCode136042_Cu.cif')
Cu_properties_ds = Cu.set_properties('ext')
Cu_properties_large_grid_ds = Cu.set_properties('large')
egrid_ext = Cu.egrids['ext']

def kkr(egrid, eps_imag, cshift=1e-5):
    eps_imag = np.array(eps_imag)
    cshift = complex(0, cshift)

    w_i = egrid
    de  = egrid[1] - egrid[0]
    
    def integration_element(w_r):
        factor = w_i / (w_i**2 - w_r**2 + cshift)
        total = np.sum(eps_imag * factor, axis=0)
        return - total * (2/np.pi) * de
        
    return np.real([integration_element(w_r) for w_r in w_i])

def f2_from_mu(mu, egrid):
    wavelength_nm_grid = hc_nm_eV / egrid
    f2 = mu * 1e-3 / (2 * re_nm * wavelength_nm_grid * Cu_n_nm3)
    correction_absorption_other = np.interp(egrid[0], Cu_properties_ds['energy'], Cu_properties_ds['f2']["element" == 'Cu']) - f2[0]
    f2 = f2 + correction_absorption_other
    return f2
    
def f1_from_f2(f2, egrid, is_plot = False, KKR_cshift = 1e-4):
    ################### combine with data in extended region ###################
    pos_below = Cu_properties_large_grid_ds['energy'] < egrid[0]
    grid_below = Cu_properties_large_grid_ds['energy'].where(pos_below, drop = True)
    values_below = np.asarray(Cu_properties_large_grid_ds['f2']["element" == 'Cu'].where(pos_below, drop = True))

    pos_above = Cu_properties_large_grid_ds['energy'] > egrid[-1]
    grid_above = Cu_properties_large_grid_ds['energy'].where(pos_above, drop = True)
    values_above = np.asarray(Cu_properties_large_grid_ds['f2']["element" == 'Cu'].where(pos_above, drop = True))
    values_above = values_above - values_above[0] + f2[-1]
    
    # Combine grids into a single array
    combined_grid = np.concatenate([grid_below, egrid, grid_above])
    # Find the smallest step size across the combined grid
    min_step = np.min(np.diff(combined_grid))
    # Create a uniform grid spanning the full range using the smallest step
    uniform_grid = np.arange(np.min(combined_grid), np.max(combined_grid) + min_step, step=min_step)
    # Interpolate values_below, tmp_f2, and values_above to the uniform grid
    interp_below = interp1d(grid_below, values_below, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_f2 = interp1d(egrid, f2, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_above = interp1d(grid_above, values_above, kind='linear', bounds_error=False, fill_value="extrapolate")
    # Initialize a zero array for the interpolated values on the uniform grid
    uniform_values = np.zeros_like(uniform_grid)
    # Apply the interpolations for each range
    # Use interpolated values for uniform grid where they exist within each segment
    # We only fill from the start to the end of each segment to avoid 0 fill between grids
    # For below egrid_ext
    mask_below = uniform_grid < egrid_ext[0]
    uniform_values[mask_below] = interp_below(uniform_grid[mask_below])
    # For egrid_ext
    mask_extended = (uniform_grid >= egrid[0]) & (uniform_grid <= egrid[-1])
    uniform_values[mask_extended] = interp_f2(uniform_grid[mask_extended])
    # For above egrid_ext
    mask_above = uniform_grid > egrid_ext[-1]
    uniform_values[mask_above] = interp_above(uniform_grid[mask_above])

    if is_plot:
        # Plot the results
        plt.figure()
        plt.plot(egrid_ext, f2, label="tmp_f2")
        plt.plot(grid_below, values_below, label="values_below")
        plt.plot(grid_above, values_above, label="values_above")
        plt.plot(uniform_grid, uniform_values, label="Uniform grid combined", linestyle="--")
        plt.xlabel("Energy (eV)")
        plt.ylabel("f2")
        plt.grid(True)
        plt.legend()
        plt.show()
        
    ################### perform Kramers-Kroning transformation ###################
    f1 = kkr(uniform_grid, uniform_values, cshift = KKR_cshift)
    
    ################### shift by a constatnt to agree with xraydb that implicitly accounts for values due to other edges ###################
    e_lower_xraydb = float(Cu_properties_ds['energy'][0])
    f1_e_lower_xraydb = float(Cu_properties_ds['f1']["element" == 'Cu'][0])

    f1_e_lower_KKR = np.interp(e_lower_xraydb, uniform_grid, f1)

    f1_shift = f1_e_lower_xraydb - f1_e_lower_KKR

    return np.interp(egrid, uniform_grid, f1 + f1_shift), f1_shift

def f1_f2_from_mu(mu, egrid, KKR_cshift = 1.e-3, is_return_shift = False):
    f2 = f2_from_mu(mu, egrid)
    f1, f1_shift = f1_from_f2(f2, egrid, is_plot = False, KKR_cshift = KKR_cshift)
    if is_return_shift:
        return f1, f2, f1_shift
    else:
        return f1, f2