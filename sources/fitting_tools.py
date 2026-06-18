import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy.special import erf
from tqdm import tqdm

# GAUSSIAN LINESHAPE
def gaussian_model(params, x, data=None):
    """
Gaussian model used for fitting. Can either return residuals, when 'data' is passed, or the evaluation of the model.
    """
    model = params['A'] * np.exp( -(x-params['mu'])**2 / (2*params['sigma']**2) ) + params['offset']
    if data is None:
        return model
    return model - data

def fit_gaussian(x_signal, y_signal, sigma=None):
    """
Perform gaussian fit of np.Array 'y_signal' using 'x_signal' as coordinate.
INPUT:
    - x_signal (np.Array): coordinate values.
    - y_signal (np.Array): values to be fitted.
    - sigma: initital guess for sigma. 
OUTPUT:
    - results.params (lmfit MinimizerResult): optmizied parameters of the model
    - results.residuals (lmfit MinimizerResult): residuals evaluated at optmized paramenters
    """
    x = x_signal
    y = y_signal
    
    if sigma is None:
        sigma = np.std(x)
    
    params = lmfit.Parameters()
    params.add('A', np.max(y)-np.min(y))
    params.add('mu', x[np.argmax(y)])
    params.add('sigma', sigma, min=0)
    params.add('offset', np.min(y))

    results = lmfit.minimize(gaussian_model, params, args=(x,), kws={'data': y})

    return results.params, results.residual

# SINUSOIDAL LINESHAPE
def sinusoidal_model(params, x, data=None):
    """
Sinusoidal model used for fitting. Can either return residuals, when 'data' is passed, or the evaluation of the model.
    """
    model = params['A'] * np.sin(x * params['f'] + params['phi']) + params['offset']
    
    if data is None:
        return model
    return model - data

def fit_sinusoidal(x_signal, y_signal, frequency=None, sampling_rate=None):
    """
Perform sinusoidal fit of np.Array 'y_signal' using 'x_signal' as coordinate. If frequency is not provided the inital guess is computed via fourier transform.
INPUT:
    - x_signal (np.Array): coordinate values.
    - y_signal (np.Array): values to be fitted.
    - frequency (float): approximate frequency of the signal
    - sampling_rate (float): approximate sampling rate of the signal
OUTPUT:
    - results (lmfit MinimizerResult): optmizied paramenters of the model
    """
    x = x_signal
    y = y_signal
    
    if frequency is None:
        if sampling_rate is None:
            sampling_rate = 10
        
        # Compute the FFT of the signal
        y_fft = np.fft.fft(y)

        # Compute the corresponding frequencies
        frequencies = np.fft.fftfreq(len(y), d=1/sampling_rate)

        # Find the index of the maximum magnitude in the FFT
        index_of_max = np.argmax(np.abs(y_fft))

        # Get the frequency corresponding to this index
        approximate_frequency = abs(frequencies[index_of_max])

        frequency = approximate_frequency

    params = lmfit.Parameters()
    params.add('A', 0.5*np.abs(np.max(y)-np.min(y)), min=0)
    params.add('f', frequency, min=0)
    params.add('phi', 0)
    params.add('offset', 0.5*(np.max(y)+np.min(y)))

    results = lmfit.minimize(sinusoidal_model, params, args=(x,), kws={'data': y})

    return results.params

# DOUBLE-EXPONENTIAL LINESHAPE
def double_exp_model(params, x, data=None):
    """
Double-exponential model used for fitting. Can either return residuals, when 'data' is passed, or the evaluation of the model.
    """
    model = params['C'] * ((params['A1'] * np.exp( - x / params['tau1'] )) + (params['A2'] * np.exp( - x / params['tau2'] )))**2 + params['offset']
    if data is None:
        return model
    return model - data


# [A*erf(t-t0)(B*exp1+C*exp2)]*2

def fit_double_exp(x_signal, y_signal, initial_guess):
    """
Perform double exponential fit of np.Array 'y_signal' using 'x_signal' as coordinate.
INPUT:
    - x_signal (np.Array): coordinate values.
    - y_signal (np.Array): values to be fitted.
    - initial_guess (list): initial guess for the parameters in the form [A1, tau1, A2, tau2].
OUTPUT:
    - results (lmfit MinimizerResult): optmizied paramenters of the model
    """
    x = x_signal
    y = y_signal

    params = lmfit.Parameters()
    params.add('C', initial_guess[4])
    params.add('A1', initial_guess[0], min=0)
    params.add('tau1', initial_guess[1], min=0)
    params.add('A2', initial_guess[2], min=0)
    params.add('tau2', initial_guess[3], min=0)
    params.add('offset', np.mean(y[-10:]))

    results = lmfit.minimize(double_exp_model, params, args=(x,), kws={'data': y})

    return results.params

# DOUBLE-GAUSSIAN LINESHAPE
def double_gaussian_model(params, x, data=None):
    """
Gaussian model used for fitting. Can either return residuals, when 'data' is passed, or the evaluation of the model.
    """
    model = (params['A1'] * np.exp( -(x-params['mu1'])**2 / (2*params['sigma1']**2) )) + (params['A2'] * np.exp( -(x-params['mu2'])**2 / (2*params['sigma2']**2))) + params['offset']
    if data is None:
        return model
    return model - data

def fit_double_gaussian(x_signal, y_signal):
    """
Perform gaussian fit of np.Array 'y_signal' using 'x_signal' as coordinate.
INPUT:
    - x_signal (np.Array): coordinate values.
    - y_signal (np.Array): values to be fitted.
OUTPUT:
    - results (lmfit MinimizerResult): optmizied parameters of the model
    """
    x = x_signal
    y = y_signal

    params = lmfit.Parameters()
    params.add('A1', np.max(y)-np.min(y))
    params.add('mu1', 941, max=945)
    params.add('sigma1', 2, min=0)
    params.add('A2', np.max(y)-np.min(y))
    params.add('mu2', 949, min=945)
    params.add('sigma2', 2, min=0)
    params.add('offset', np.min(y))

    results = lmfit.minimize(double_gaussian_model, params, args=(x,), kws={'data': y})

    return results.params, results.residual