import copy
from icecream import ic
import numpy as np
from numba import jit
import scipy.constants as spconst
from tqdm.notebook import tqdm
import xraydb

from TASsello import f1f2Sim

"""
09.01.2025 - 14:29
v3 includes the SiO2 bottom layer

10.01.2025 - 11:09
v4_fit includes a parameter "f_eV_shift" used to shift the energy scale of the scattering factors computed with FEFF for Cu.
This factor id by default set to -0.11332 eV, which is the value obtained by the optimization.

20.01.2025 - 11:22
v5_fit includes the new parameter hw_ERES, which is the resonant central energy of the ERES contribution

28.01.2025 - 16:43
v6_FEFF includes the scattering factor from FEFF. These are calculated with the TASsello library, Epump and t can be used as inputs.

13.02.2025 - 16:04
v7_FEFF is identical to v6 but includes some code optimizations:
    - caches values for the ML parameter to avoid reloads at every _get_value()
    - jit compiled interpolate function
    
25.02.2025 - 12:20
v8 vectorizes the calculation along energy.
 """

@jit(nopython=True)
def interpolate(xOut, xIn, yIn):
    yOut = np.interp(xOut, xIn, yIn)
    return yOut

@jit(nopython=True)
def Xloops(alphagrid, wlgrid, tML, uj, rj):
    nlayers = len(tML)-1
    
    X = np.zeros((alphagrid, wlgrid), dtype=np.complex128)
    for i in range(nlayers-1, 0, -1):
        exp_factor = np.exp( -2j * uj[i-1] * tML[i-1] )
        num = rj[i] + X 
        den = 1 + rj[i] * X
        X = exp_factor * num / den
    numf = rj[0] + X
    denf = 1 + rj[0] * X
    X = numf / denf
    
    return np.abs(X)**2

re = 1e9 * spconst.value('classical electron radius') #nm

wl_L3_nm = 1.3293 #nm
wl_Lalpha_eV = 929.7 #eV

MB = 10.81
MC = 12.01
MSi = 28.09
MCu = 63.55

class Sample:
    dirname_f = './scattering_factors/'
    dirname_params = './fit_parameters/'
    
    def __init__(self, samplename, wl_eV=None, f_eV_shift=None):
        self.filename_params = f'{self.dirname_params}Params_{samplename}'

        self.MLrep = np.loadtxt(self.filename_params, dtype=str, skiprows=1, max_rows=1)[1].astype(int)
        self.nlayers = self.MLrep*3 + 1

        if samplename[0] == 'M':
            self.sequence = 'direct'
        elif samplename[0] == 'W':
            self.sequence = 'inverse'
        
        if wl_eV is not None:
            self.set_wavelength(wl_eV)

        if f_eV_shift is None:
            self.set_f_eV_shift(-0.11332)
        elif isinstance(f_eV_shift, (int, float)):
            self.set_f_eV_shift(f_eV_shift)

        self.f1f2Simulator = f1f2Sim()

    def _load_parameters(self):
        if not hasattr(self, '_parameters_cache'):
            self._parameters_cache = np.loadtxt(self.filename_params, dtype=str, skiprows=2, usecols=(0))
            self._values_cache = np.loadtxt(self.filename_params, dtype=float, skiprows=2, usecols=(1))
        
    def _get_value(self, parameter, layer):
        self._load_parameters() # Load only once
        param_name = f'{layer}.set{parameter}'
        if parameter == 'period':
            param_name = 'cp.setPeriodML'
            
        return 1e-1*self._values_cache[np.nonzero(self._parameters_cache==param_name)]
        
    def set_wavelength(self, wls_eV):
        """
        Set the wavelengths array in eV and nm. Then also sets the k0 array.
        Input:
            - wls_eV (np.array)
        """
        self.wl_eV = wls_eV
        self.wl_nm = 1e9 * spconst.h * spconst.c / wls_eV / spconst.eV
        self.k0 = 2 * np.pi / self.wl_nm        

    def set_q(self, q):
        self.q = q

    def set_q_from_alpha(self, alpha):
        self.q = 2 * self.k0 * np.sin(np.deg2rad(alpha))

    def set_f_eV_shift(self, f_eV_shift):
        self.f_eV_shift = f_eV_shift

    def set_f1_f2_Cu(self):
        self.f1_Cu, self.f2_Cu = self.f1f2Simulator.get_f1_f2()
       
    def get_q(self):
        return self.q

    def get_wavelength_eV(self):
        return self.wl_eV

    def get_k0(self):
        return self.k0

    def get_nlayers(self):
        return self.nlayers

    def get_scattering_factor(self, element):        
        if element == 'Cu':
            f = interpolate(self.wl_eV, self.f1f2Simulator.egrid + self.f_eV_shift, self.f1_Cu) - (1j * interpolate(self.wl_eV, self.f1f2Simulator.egrid + self.f_eV_shift, self.f2_Cu))
            return np.broadcast_to(f, np.shape(self.q))
        else:
            f1 = xraydb.f1_chantler(element, self.wl_eV) + xraydb.f0(element, self.q/(4*np.pi))
            f2 = xraydb.f2_chantler(element, self.wl_eV) # flip sign to be consistent with the old database values used
            return f1 - 1j*f2
    
    def get_chi0_layer(self, layer):
        density = 10 * self._get_value('Dens', layer) #10 is to match the units
        Veff = 1e-3 / density # FU/nm**3
        
        if layer == 'B4C':
            f_layer = ( 4*self.get_scattering_factor('B') + self.get_scattering_factor('C') ) / Veff
        elif layer == 'Cu':
            f_layer = self.get_scattering_factor('Cu') / Veff
        elif layer == 'SiC':
            f_layer = ( self.get_scattering_factor('Si') + self.get_scattering_factor('C') ) / Veff
        elif layer == 'bottom_SiO2':
            f_layer = ( self.get_scattering_factor('Si') + 2*self.get_scattering_factor('O') ) / Veff

        return - ( re * wl_L3_nm**2 / np.pi ) * f_layer

    def create_ML_thickness(self):       
        tSi = np.asarray([0])
        tSiO2 = self._get_value('D', 'bottom_SiO2')
        tlayer = np.concatenate((
                             self._get_value('D', 'B4C'),
                             self._get_value('D', 'Cu'),
                             self._get_value('period', None)-self._get_value('D','Cu')-self._get_value('D','B4C')
                                ))
        if self.sequence == 'inverse':
            tlayer = np.flip(tlayer)
        return np.concatenate((np.tile(tlayer, self.MLrep), tSiO2, tSi))
        
    def create_ML_roughness(self):
        sigmaSi = self._get_value('Sigma', 'Sub')
        sigmaSiO2 = self._get_value('Sigma', 'bottom_SiO2')
        sigmalayer = np.concatenate((
                                 self._get_value('Sigma', 'B4C'),
                                 self._get_value('Sigma', 'Cu'),
                                 self._get_value('Sigma', 'SiC')
                               ))
        if self.sequence == 'inverse':
            sigmalayer = np.flip(sigmalayer)
        return np.concatenate((np.tile(sigmalayer, self.MLrep), sigmaSiO2, sigmaSi))

    def create_ML_chi0(self):        
        chi0Si = np.broadcast_to([-0.1033e-2 + 1j * 0.92017e-4], np.shape(self.q)) #substrate
        chi0SiO2 = np.broadcast_to(self.get_chi0_layer('bottom_SiO2'), np.shape(self.q)) #SiO2
        chi0layer = np.stack((
                              self.get_chi0_layer('B4C'),
                              self.get_chi0_layer('Cu'),
                              self.get_chi0_layer('SiC')
                            ))
        if self.sequence == 'inverse':
            chi0layer = np.flip(chi0layer, axis=0)
        self.chi0 = np.concatenate(
                                    (np.tile(chi0layer, (self.MLrep, 1, 1)),
                                     chi0SiO2[np.newaxis, ...],
                                     chi0Si[np.newaxis, ...]
                                    ), axis=0)
        return self.chi0

    def copy(self):
        # Return a deep copy of the instance
        return copy.deepcopy(self)

class ParrattSimulator:
    def __init__(self, sample, alpha=None, alphagrid=2000, wls_eV=np.linspace(915, 945, 200)):
        self.sample = sample
        self.nlayers = self.sample.get_nlayers()

        if alpha is None:
            self.alphagrid = alphagrid
            self.alpha = np.linspace(0, 45, alphagrid) #deg
        elif alpha is not None:
            self.alpha = alpha
            self.alphagrid = len(alpha)
        
        self.set_wavelengths(wls_eV)
        
    def set_wavelengths(self, wls_eV):
        if not isinstance(wls_eV, np.ndarray):
            wls_eV = np.array(wls_eV)
        
        self.wl_eV_list = wls_eV
        self.sample.set_wavelength(wls_eV)
        
        wl_nm = 1e9 * spconst.h * spconst.c / wls_eV / spconst.eV
        k0 = 2 * np.pi / wl_nm

        self.q_ak = 2 * np.outer(np.sin(np.deg2rad(self.alpha)), k0)
        self.sample.set_q(self.q_ak)
        
    def create_u_vectors(self, xvar):        
        if xvar is None:
            xvar = 'deg'
        
        if xvar not in ['deg', 'q']:
            raise ValueError(f"Invalid choice: '{xvar}'. Allowed values are 'deg' or 'q'.")
        
        angle = np.deg2rad(self.alpha)[..., np.newaxis]
        chi0 = self.sample.create_ML_chi0()
        k0 = self.sample.get_k0()[np.newaxis, ...]
        
        uj = np.zeros((self.nlayers, self.alphagrid, len(self.wl_eV_list)), dtype=complex)
        if xvar == 'deg':
            u0 = - k0 * np.sin(angle)
            uj = - k0 * np.sqrt( np.sin(angle)**2 + chi0)
        if xvar == 'q':
            u0 = - 0.5 * self.sample.get_q()
            uj = - np.sqrt( (0.25 * self.sample.get_q()**2) + (k0**2 * chi0) )            
            
        return u0, uj
    
    def simulate(self, xvar=None):
        if not hasattr(self.sample, 'q'):
            self.sample.set_q_from_alpha(self.alpha)
            
        tML = self.sample.create_ML_thickness()
        sigmaML = self.sample.create_ML_roughness()

        u0, uj = self.create_u_vectors(xvar)
    
        rj = np.zeros((np.size(tML), self.alphagrid, len(self.wl_eV_list)), dtype=np.complex128)
        rj[0] = (( u0 - uj[0] ) / ( u0 + uj[0] )) * np.exp( -2 * sigmaML[0]**2 * u0 * uj[0] )
        rj[1:self.nlayers] = ((uj[:self.nlayers-1] - uj[1:self.nlayers]) / (uj[:self.nlayers-1] + uj[1:self.nlayers])) * \
                                np.exp(-2 * sigmaML[1:self.nlayers, np.newaxis, np.newaxis]**2 * uj[:self.nlayers-1] * uj[1:self.nlayers])
        
        return Xloops(self.alphagrid, len(self.wl_eV_list), tML, uj, rj) #np.abs(X)**2