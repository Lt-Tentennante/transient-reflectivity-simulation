# TASsello
TASsello (Transient Atomic Scattering - sello) is a small library to compute the transient atomic scattering factor of atoms following XFEL irradiation. The library uses as inputs the result from the Boltzmann and the FEFF code. In short, Boltzmann computes the relative abundances of charged states. FEFF uses this input to compute the absorption properties of this population. TASsello takes the results of FEFF and generates the atomic scattering factor.

The input parameters of TASsello is only the X-Ray field with the dimension (x,y,t), and the output are the real and imaginary part of the atomic scattering factor. The notebook runs parallely on each field point. The user can set up the space and time grid to refine the calculation.

## Example folder

### TASsello_example.ipynb
A minimal notebook that provides an example on how to use the TASsello library. 