import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import os
file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

from .constants import cm, conversion_A2_into_umm1, fs, um
from .functions import egrid_ext, kkr, f1_f2_from_mu 

# LOADING DATA
# Directories and data loading
used_data_directory = directory_path + '/data_dependencies/used_data/'
SCF_grid_folder = used_data_directory + 'SCF_Grid_Results_Shift_d-Centroid/'
Boltzmann_folder = used_data_directory + '/Boltzmann_configuration_probabilities_data/'

# Load data
Epump_grid_interp = np.loadtxt(Boltzmann_folder + 'p_grid_interp.txt')
conf_prob_regrid_t_p__ip_t_ii = np.load(Boltzmann_folder + 'conf_prob_regrid_t_p__ip_t_ii.npy')
t_grid_global = np.loadtxt(Boltzmann_folder + 't_grid_global.txt')
p_vs_Te__p = np.loadtxt(Boltzmann_folder + 'p_vs_Te__p.txt')
p_vs_Te__Te = np.loadtxt(Boltzmann_folder + 'p_vs_Te__Te.txt')
Te_grid = np.loadtxt(Boltzmann_folder + 'Te_grid.txt')
Te_names = np.loadtxt(Boltzmann_folder + 'Te_names.txt', dtype=str, delimiter=',')
shifts__N3d_iTe = np.loadtxt(Boltzmann_folder + 'shifts__N3d_iTe.txt')

# Constants and configuration
n3d = 10
ar3dN = np.array([10, 9, 8, 7, 6, 5, 4, 3])

# Predefined arrays
d3occ = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
N3d_conf = d3occ
ii_max = 10
mu_interp2D_vs_Te_e___m_ii = ['not specified'] * (ii_max + 1)

# Interpolation for Te vs p
Te_vs_p = interp1d(p_vs_Te__p, p_vs_Te__Te, kind='linear', fill_value=(25.0, 0.025), bounds_error=False)

# Load SCF data and interpolate for each N3d configuration
for ii in range(len(ar3dN)):

    xanes_values_m = []
    N3d = ar3dN[ii]

    for iTe, Te_name in enumerate(Te_names):
        CuXANES_d = np.loadtxt(f"{SCF_grid_folder}{Te_name}/CuXANES_{N3d}.dat")
        if CuXANES_d.size:
            egrid_local, xanes = CuXANES_d[:, 0], CuXANES_d[:, 3] * conversion_A2_into_umm1
            xanes_interp = interp1d(egrid_local, xanes, kind='quadratic', bounds_error=False, fill_value=0)
            mu = np.asarray([xanes_interp(e + shifts__N3d_iTe[ii, iTe]) for e in egrid_ext])
            xanes_values_m.append(mu)

    if xanes_values_m:
        mu_interp2D_vs_Te_e___m_ii[ii] = RegularGridInterpolator((Te_grid, egrid_ext), xanes_values_m, bounds_error=False, fill_value=0)
        
# Spatial profile used in Beata's calculations
FWHM = 15
w0 = 3.5 * um
conversion_Epump_Intensity = 2 / (np.pi * w0**2 * 1.064 * FWHM * fs) * cm**2

def convert_E_to_I(E, w0, t_FWHM):
    """
    Convert the pulse energy to pulse intensity. Input parameters are in: [E]=J, [w0]=um (FWHM), t_FWHM=[fs]
    """
    w0 = w0/2 * um
    conversion_factor = 2 / (np.pi * w0**2 * 1.064 * t_FWHM * fs) * cm**2
    return E * conversion_factor

def mu_weighted_e(I_pump, t, egrid_obs):
    # Converting I_pump (W/cm**2) to E_pump (J) 
    E_pump = I_pump / conversion_Epump_Intensity
    
    # getting configuration probabilities for given T and Epump
    iEpump = (np.abs(Epump_grid_interp - E_pump)).argmin()
    probs_it_ii = conf_prob_regrid_t_p__ip_t_ii[iEpump]
    probs_it_ii = np.nan_to_num(probs_it_ii)
    it = (np.abs(t_grid_global - t)).argmin()
    probs_ii = probs_it_ii[it]
    
    # Te vs t from average 3d occupation
    p3d_av_it = np.einsum('ti,i->t', probs_it_ii, d3occ) / n3d
    Te_t_av = interp1d(t_grid_global, Te_vs_p(p3d_av_it), kind='linear', bounds_error=False)
    Te = Te_t_av(t)
    
    mu_weighted = np.zeros(len(egrid_obs))

    for ii in range(len(ar3dN)):
        prob = probs_ii[ii]
        
        Te_mesh, E_mesh = np.meshgrid([Te], egrid_obs, indexing='ij')
        mu_xanes = mu_interp2D_vs_Te_e___m_ii[ii]((Te_mesh, E_mesh))[0]

        mu_weighted += prob * mu_xanes

    return mu_weighted

class f1f2Sim:
    def __init__(self):
        self.set_tprofile()
        self.egrid = egrid_ext

    def set_tprofile(self):
        self.t0 = 30
        self.FWHM = 15
        self.sigma = FWHM / 2.3548

    def set_sim_params(self, I, t):
        """
        Set the simulation parameters.

        Parameters:
        - I (float): Intensity of the pump pulse in W/cm^2.
        - t (float): Time within the pulse.
        """
        self.Ipump = I
        self.t = t

    def get_mu(self):
        return mu_weighted_e(self.Ipump, self.t, self.egrid)

    def get_f1_f2(self, is_return_shift=False):
        return f1_f2_from_mu(self.get_mu(), self.egrid, is_return_shift=is_return_shift)

    def I_pump_t(self, t):
        return (1/np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(-(t-self.t0)**2 / (2 * self.sigma**2))

    def plot_I_pump(self, t=np.linspace(0,60,60)):
        plt.figure()
        plt.plot(t, self.I_pump_t(t) / self.I_pump_t(self.t0))
        plt.plot(np.linspace(self.t0 - self.FWHM/2, self.t0 + self.FWHM/2, 50), 0.5*np.ones(50))
        plt.ylabel('I / Imax')
        plt.xlabel('t (fs)')
        plt.grid(True)
        plt.show()