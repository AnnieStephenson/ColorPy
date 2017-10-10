"""
Created on Wed Dec 28 15:45:34 2016

functions for calculating the color values in various color spaces from a reflectance spectrum

.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
"""

import numpy as np
import matplotlib.pyplot as plt
import colorpy.ciexyz, colorpy.illuminants, colorpy.colormodels
from scipy import interpolate

def color_from_refl(refl, wavelengths = np.arange(360, 831), illuminant_name = 'D65', show_spectrum_plot = False):
    """
    Calculate the color values in various color spaces from a reflectance spectrum. 

    Parameters
    ----------
    refl: 1D numpy array
        reflectance values corresponding to wavelengths
    wavelengths: 1D numpy array
        wavelengths corresponding to reflectance values
    illuninant_name : string
        any illuminant described in colorpy.illuminants. Default is set to D65 
        which simulates normal sunlight conditions
    show_spectrum_plot: boolean
        determines whether to plot power spectrum plot and color swatch. Default
        set to False. 
    
    Returns
    -------
    color: dictionary
        keys: 'irgb', 'rgb', 'xyz', 'lab'
        values: irgb, rgb, xyz, and lab colorspace values in the form of a 1x3 array

    Notes
    -----    
    Entries in colorpy.illuminants are defined only for wavelengths 360-830 nm.
    Therefore, if reflectance is given outside this range, the code will return
    an error

    The illuminant (light source) is "D65" (standard daylight) by default,
    which approximates natural daylight. You can use any other function in 
    colorpy.illuminants instead if you like, or make your own array with the 
    same format.

    """
    if illuminant_name == 'D65':
        illuminant = colorpy.illuminants.get_illuminant_D65()
    else:
        raise ValueError("That illuminant hasn't been added...yet. Use D65 for now")
    
    assert len(refl) == len(wavelengths), 'expecting reflectance of length equal to wavelength length'
    assert all(0 <= x <= 1 for x in refl), 'expecting reflectance less than or equal to 1'
    assert min(wavelengths) >= 360, 'expecting reflectance for wavelength values > 360 nm'
    assert max(wavelengths) <=830, 'expecting reflectance for wavelength values < 830 nm'
    
    f_illuminant = interpolate.interp1d(illuminant[:,0], illuminant[:,1])
    illum = f_illuminant(wavelengths)

    # multiply illuminant power by reflectance to find the power of the reflected light
    refl_power = illum*refl
    spectrum = np.transpose(np.vstack((wavelengths, refl_power)))

    # plots the power spectrum of reflected light vs wavelength
    if show_spectrum_plot:
        plt.figure()
        colorpy.plots.spectrum_plot(spectrum,
                                    title='Reflected light',
                                    filename='temp.png',
                                    ylabel='Power')	
    
    xyz = colorpy.ciexyz.xyz_from_spectrum(spectrum)
    lab = colorpy.colormodels.lab_from_xyz(xyz)
    luv = colorpy.colormodels.luv_from_xyz(xyz)
    rgb = colorpy.colormodels.rgb_from_xyz(xyz)
    irgb = colorpy.colormodels.irgb_from_rgb(rgb)
    color = {'rgb': rgb, 'irgb': irgb, 'xyz': xyz, 'lab': lab, 'luv': luv}
    return color
    

def test():
    """
    test to make sure color is:
    - grey when all reflectance values 0.5
    - same grey when skipping
    - white when all reflectance values 1
    - black when all reflectance values 0
    """
    refl = [0.5 for i in range(360,831)]
    color = color_from_refl(refl, show_spectrum_plot=True)
    print(color['rgb'])
 
    wavelengths = np.arange(360,830,10)
    refl = [0.5 for i in wavelengths]
    color = color_from_refl(refl, wavelengths, show_spectrum_plot=True)
    print(color['rgb'])

    refl = [1.0 for i in range(360,831)]
    color = color_from_refl(refl, show_spectrum_plot=True)
    print(color['rgb'])
 
    refl = [0 for i in range(360,831)]
    color = color_from_refl(refl, show_spectrum_plot=True)
    print(color['rgb'])
