import scipy.constants as sp_const
from scipy.constants import physical_constants

# CONSTANTS
amu = sp_const.value('atomic mass constant')
h = sp_const.h # Planck constant (h) in Joules seconds
h_eV = h / sp_const.e # Convert h to electron volts (eV) using the conversion factor from Joules to eV (1 eV = 1.602176634 × 10^-19 Joules)
c = sp_const.c # Speed of light (c) in meters per second
c_nm_per_second = c * 1e9 # Convert c to nm per second (1 Angstrom = 1 × 10^-9 meters)
re_nm = physical_constants['classical electron radius'][0] * 1e9
hc_nm_eV = h_eV * c_nm_per_second # Calculate h * c in nm eV

# Conversions
gramm = 1e-3
A  = 1e-10
nm = 1e-9
um = 1e-6
cm = 1e-2
fs = 1e-15
Cu_density = 8.96 #g/cm3
Cu_A = 63.546
Cu_n = Cu_density * (gramm/(cm**3)) / (Cu_A * amu)
Cu_n_nm3 = Cu_n * nm**3
conversion_A2_into_umm1 = A**2 * Cu_n * um