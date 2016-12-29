# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:45:34 2016

@author: Annie Stephenson
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import colorpy, colorpy.ciexyz, colorpy.illuminants, colorpy.colormodels

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
    Therefore, if reflectance is given outside this range, the smaller of the
    two vectors will be padded with zeros to ensure they end up the same length.
    Reflectance values must range from 0 to 1.

    The illuminant (light source) is "D65" (standard daylight) by default,
    which approximates natural daylight. You can use any other function in 
    colorpy.illuminants instead if you like, or make your own array with the same format.

    """
    if illuminant_name == 'D65':
        illuminant = colorpy.illuminants.get_illuminant_D65()
    else:
        raise ValueError("That illuminant hasn't been added...yet. Use D65 for now")
    
    assert len(refl) == len(wavelengths), 'expecting reflectance of length equal to wavelength length'
    assert all(0 <= x <= 1 for x in refl), 'expecting reflectance less than or equal to 1'
    assert min(wavelengths) >= 360, 'expecting reflectance for wavelength values > 360 nm'
    assert max(wavelengths) <=830, 'expecting reflectance for wavelength values < 830 nm'

    wavelengths = np.intersect1d(illuminant[:,0], wavelengths)
    illum = illuminant[:,0].tolist()
    wl_index = []
    for i in range(0,len(wavelengths)):
        wl_index.append(illum.index(wavelengths[i]))
    illuminant_crop = []
    for i in wl_index:
        illuminant_crop.append(illuminant[i,:])
    illuminant = np.array(illuminant_crop)

    # illuminant is an 2D array where the first column is wavelength and the 
    # second is power. We multiply that power by reflectance to find the power 
    # of the reflected light
    refl_power = illuminant[:,1]*refl
    spectrum = np.transpose(np.vstack((wavelengths, refl_power)))

    if show_spectrum_plot:
        plt.figure()
        colorpy.plots.spectrum_plot(spectrum,
                                    title='Reflected light',
                                    filename='temp.png',
                                    ylabel='Power')		# plots the power spectrum of reflected light vs wavelength
    
    xyz = colorpy.ciexyz.xyz_from_spectrum(spectrum)
    lab = colorpy.colormodels.lab_from_xyz(xyz) 
    rgb = colorpy.colormodels.rgb_from_xyz(xyz)
    irgb = colorpy.colormodels.irgb_from_rgb(rgb)
    color = {'rgb': rgb, 'irgb': irgb, 'xyz': xyz, 'lab': lab}
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