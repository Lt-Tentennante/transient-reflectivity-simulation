import copy
from icecream import ic
import numpy as np
import scipy.constants as spconst
from tqdm.notebook import tqdm
import xraydb

"""
09.01.2025 - 14:29
v3 includes the SiO2 bottom layer

10.01.2025 - 11:09
v4_fit includes a parameter "f_eV_shift" used to shift the energy scale of the scattering factors computed with FEFF for Cu.
This factor id by default set to -0.11332 eV, which is the value obtained by the optimization.

20.01.2025 - 11:22
v5_fit includes the new parameter hw_ERES, which is the resonant central energy of the ERES contribution
 """

re = 1e9 * spconst.value('classical electron radius') #nm

wl_L3_nm = 1.3293 #nm
wl_Lalpha_eV = 929.7 #eV

MB = 10.81
MC = 12.01
MSi = 28.09
MCu = 63.55

class Sample:
    dirname_f = './sources/XRR/scattering_factors/'
    dirname_params = './sources/XRR/fit_parameters/'
    
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
        
    def _get_value(self, parameter, layer):
        param_name = f'{layer}.set{parameter}'
        if parameter == 'period':
            param_name = 'cp.setPeriodML'
        
        parameters = np.loadtxt(self.filename_params, dtype=str, skiprows=2, usecols=(0))
        values = np.loadtxt(self.filename_params, dtype=float, skiprows=2, usecols=(1))
    
        return 1e-1*values[np.nonzero(parameters==param_name)]
    
    def set_wavelength(self, wl_eV):
        self.wl_eV = wl_eV
        self.wl_nm = 1e9 * spconst.h * spconst.c / wl_eV / spconst.eV
        self.k0 = 2 * np.pi / self.wl_nm

    def set_q(self, q):
        self.q = q

    def set_q_from_alpha(self, alpha):
        self.q = 2 * self.k0 * np.sin(np.deg2rad(alpha))

    def set_f_eV_shift(self, f_eV_shift):
        self.f_eV_shift = f_eV_shift
            
    def set_ERES(self, f_ERES, gamma2=1.6, hw_ERES=929.7):
        self.f_ERES = f_ERES # Amplitude
        self.gamma2 = gamma2 # Width
        self.hw_ERES = hw_ERES # Central wavelength

    def get_q(self):
        return self.q

    def get_wavelength_eV(self):
        return self.wl_eV

    def get_k0(self):
        return self.k0

    def get_nlayers(self):
        return self.nlayers

    def get_scattering_factor(self, element, return_res=False):
        def f_resonant(x):
            # return self.f_ERES * self.gamma2 * ( ((x - self.hw_ERES) - 1j*self.gamma2) / ((x - self.hw_ERES)**2 + self.gamma2**2) )
            # return self.f_ERES * (2 / self.gamma2) * ( ((x - self.hw_ERES) - 1j*0.5*self.gamma2) / ((x - self.hw_ERES)**2 + (0.5*self.gamma2)**2) )
            return self.f_ERES * ( ((x - self.hw_ERES) - 1j*0.5*self.gamma2) / ((x - self.hw_ERES)**2 + (0.5*self.gamma2)**2) )
            
        if element == 'Cu':
            f1_filename = f'{self.dirname_f}{element}-f1-FEFF.dat'
            f2_filename = f'{self.dirname_f}{element}-f2-FEFF.dat'
            f1 = np.loadtxt(f1_filename)
            f2 = np.loadtxt(f2_filename)
            f = np.interp(self.wl_eV, f1[:,0]+self.f_eV_shift, f1[:,1]) - (1j * np.interp(self.wl_eV, f2[:,0]+self.f_eV_shift, f2[:,1])) + f_resonant(self.wl_eV)
            if return_res:
                return np.broadcast_to(f, np.shape(self.q)), f_resonant(self.wl_eV)
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
        chi0Si = np.broadcast_to([-0.1033e-2 + 1j * 0.92017e-4], np.shape(self.q)).reshape(1, -1) #substrate
        chi0SiO2 = np.broadcast_to(self.get_chi0_layer('bottom_SiO2'), np.shape(self.q)).reshape(1, -1) #SiO2
        chi0layer = np.stack((
                              self.get_chi0_layer('B4C'),
                              self.get_chi0_layer('Cu'),
                              self.get_chi0_layer('SiC')
                            ))
        if self.sequence == 'inverse':
            chi0layer = np.flip(chi0layer, axis=0)
        self.chi0 = np.concatenate((np.tile(chi0layer, (self.MLrep, 1)), chi0SiO2, chi0Si), axis=0)
        return self.chi0

    def copy(self):
        # Return a deep copy of the instance
        return copy.deepcopy(self)

class ParrattSimulator:
    def __init__(self, sample, alphagrid=2000, wls_eV=np.linspace(915, 945, 200)):
        self.sample = sample
        self.nlayers = self.sample.get_nlayers()

        self.alphagrid = alphagrid
        self.alpha = np.linspace(0, 45, alphagrid) #deg
        
        self.set_wavelengths(wls_eV)
        
    def set_wavelengths(self, wls_eV):
        self.wl_eV_list = wls_eV
        
        wl_nm = 1e9 * spconst.h * spconst.c / wls_eV / spconst.eV
        k0 = 2 * np.pi / wl_nm

        self.q_ak = 2 * np.outer(np.sin(np.deg2rad(self.alpha)), k0)

    def create_u_vectors(self, xvar):        
        if xvar is None:
            xvar = 'deg'
        
        if xvar not in ['deg', 'q']:
            raise ValueError(f"Invalid choice: '{xvar}'. Allowed values are 'deg' or 'q'.")
        
        angle = np.deg2rad(self.alpha)
        chi0 = self.sample.create_ML_chi0()
        k0 = self.sample.get_k0()
        
        uj = np.zeros((self.nlayers, self.alphagrid), dtype=complex)
        if xvar == 'deg':
            u0 = - k0 * np.sin(angle)
            for i in range(0, self.nlayers):
                uj[i] = - k0 * np.sqrt( np.sin(angle)**2 + chi0[i])
        if xvar == 'q':
            u0 = - 0.5 * self.sample.get_q()
            for i in range(0, self.nlayers):
                uj[i] = - np.sqrt( (0.25 * self.sample.get_q()**2) + (k0**2 * chi0[i]) )            
            
        return u0, uj
        
    def compute_reflectivity(self, xvar=None):
        if not hasattr(self.sample, 'q'):
            self.sample.set_q_from_alpha(self.alpha)
            
        tML = self.sample.create_ML_thickness()
        sigmaML = self.sample.create_ML_roughness()

        u0, uj = self.create_u_vectors(xvar)
        
        rj = np.zeros((np.size(tML), self.alphagrid), dtype=complex)
        for i in range(0, self.nlayers):
            if i==0:
                rj[i] = (( u0 - uj[0] ) / ( u0 + uj[0] )) * \
                            np.exp( -2 * sigmaML[0]**2 * u0 * uj[0] )
            else:
                rj[i] = (( uj[i-1] - uj[i] ) / ( uj[i-1] + uj[i] )) * \
                            np.exp( -2 * sigmaML[i]**2 * uj[i-1] * uj[i] )

        X = 0
        for i in range(self.nlayers-1, 0, -1):
            X = np.exp( -2 * 1j * uj[i-1] * tML[i-1] ) * (( rj[i] + X ) / ( 1 + rj[i] * X ))
        X = ( rj[0] + X ) / (1 + rj[0] * X)

        return np.abs(X)**2

    def simulate_multi_wl(self, xvar=None):
        R = np.zeros((self.alphagrid, len(self.wl_eV_list)))
        for i, wl_eV_ in enumerate(tqdm(self.wl_eV_list, desc='Simulating reflectivity', unit='wavelength')):
            self.sample.set_wavelength(wl_eV_)
            self.sample.set_q(self.q_ak[:,i])
            # self.sample.set_q_from_alpha(self.alpha)

            R[:,i] = self.compute_reflectivity(xvar)
        return R