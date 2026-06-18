# XRR - X-Ray Reflectivity
XRR is a minimal module to compute the X-Ray Reflectivity curve of a nanostructured multilayer (ML) sample. At its core XRR uses a recursive Parratt's algorithm to compute the reflectivity.

The module is tailored to the specific case of Cu containing ML samples. Specifically, the Cu atoms scattering factor is modified to account for the effect of the transient contribution. We developed two versions of XRR that account for this transient effect in different ways.

## XRR_2level_fit.py
In this module the scattering factor of Cu is simply taken from tabulated data. In addition to the tabulated value a resonant contribution is addedd which follows the equation:
$$
    f_T(\omega) = A \frac{(\omega - \omega_0) - i  \gamma_2/2}{(\omega - \omega_0)^2 + (\gamma_2/2)^2}.
$$
This is the contribution to the scattering factor of a 2-level system with width $\gamma_2$ and resonant energy $\omega_0$, $\omega$ is the photon energy of the incoming field.

The three parameters $A$, $\omega_0$ and $\gamma_2$ can be left free to be fitted to the experimental data - this is done in `2level_fit.ipynb`.

## XRR_self_consistent.py
in this version of the module the scattering factor of Cu atoms is calculated with with the TASsello library with the self-consistent approach.