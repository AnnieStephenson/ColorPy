'''
colormodels.py - Conversions between color models

Description:

Defines several color models, and conversions between them.

The models are:

xyz - CIE XYZ color space, based on the 1931 matching functions for a 2 degree field of view.
    Spectra are converted to xyz color values by integrating with the matching functions in ciexyz.py.

    xyz colors are often handled as absolute values, conventionally written with uppercase letters XYZ,
    or as scaled values (so that X+Y+Z = 1.0), conventionally written with lowercase letters xyz.

    This is the fundamental color model around which all others are based.

rgb - Colors expressed as red, green and blue values, in the nominal range 0.0 - 1.0.
    These are linear color values, meaning that doubling the number implies a doubling of the light intensity.
    rgb color values may be out of range (greater than 1.0, or negative), and do not account for gamma correction.
    They should not be drawn directly.

irgb - Displayable color values expressed as red, green and blue values, in the range 0 - 255.
    These have been adjusted for gamma correction, and have been clipped into the displayable range 0 - 255.
    These color values can be drawn directly.

Luv - A nearly perceptually uniform color space.

Lab - Another nearly perceptually uniform color space.

As far as I know, the Luv and Lab spaces are of similar quality.
Neither is perfect, so perhaps try each, and see what works best for your application.

The models store color values as 3-element NumPy vectors.
The values are stored as floats, except for irgb, which are stored as integers.

Constants:

SRGB_Red
SRGB_Green
SRGB_Blue
SRGB_White -
    Chromaticity values for sRGB standard display monitors.

PhosphorRed
PhosphorGreen
PhosphorBlue
PhosphorWhite -
    Chromaticity values for display used in initialization.
    These are the sRGB values by default, but other values can be chosen.

CLIP_CLAMP_TO_ZERO = 0
CLIP_ADD_WHITE     = 1
    Available color clipping methods.  Add white is the default.

Functions:

'Constructor-like' functions:

xyz_color (x, y, z = None) -
    Construct an xyz color.  If z is omitted, set it so that x+y+z = 1.0.

xyz_normalize (xyz) -
    Scale so that all values add to 1.0.
    This both modifies the passed argument and returns the normalized result.

xyz_normalize_Y1 (xyz) -
    Scale so that the y component is 1.0.
    This both modifies the passed argument and returns the normalized result.

xyz_color_from_xyY (x, y, Y) -
    Given the 'little' x,y chromaticity, and the intensity Y,
    construct an xyz color.  See Foley/Van Dam p. 581, eq. 13.21.

rgb_color (r, g, b) -
    Construct a linear rgb color from components.

irgb_color (ir, ig, ib) -
    Construct a displayable integer irgb color from components.

luv_color (L, u, v) -
    Construct a Luv color from components.

lab_color (L, a, b) -
    Construct a Lab color from components.

Conversion functions:

rgb_from_xyz (xyz) -
    Convert an xyz color to rgb.

xyz_from_rgb (rgb) -
    Convert an rgb color to xyz.

irgb_string_from_irgb (irgb) -
    Convert a displayable irgb color (0-255) into a hex string.

irgb_from_irgb_string (irgb_string) -
    Convert a color hex string (like '#AB13D2') into a displayable irgb color.

irgb_from_rgb (rgb) -
    Convert a (linear) rgb value (range 0.0 - 1.0) into a 0-255 displayable integer irgb value (range 0 - 255).

rgb_from_irgb (irgb) -
    Convert a displayable (gamma corrected) irgb value (range 0 - 255) into a linear rgb value (range 0.0 - 1.0).

irgb_string_from_rgb (rgb) -
    Clip the rgb color, convert to a displayable color, and convert to a hex string.

irgb_from_xyz (xyz) -
    Convert an xyz color directly into a displayable irgb color.

irgb_string_from_xyz (xyz) -
    Convert an xyz color directly into a displayable irgb color hex string.

luv_from_xyz (xyz) -
    Convert CIE XYZ to Luv.

xyz_from_luv (luv) -
    Convert Luv to CIE XYZ.  Inverse of luv_from_xyz().

lab_from_xyz (xyz) -
    Convert color from CIE XYZ to Lab.

xyz_from_lab (Lab) -
    Convert color from Lab to CIE XYZ.  Inverse of lab_from_xyz().

Gamma correction:

simple_gamma_invert (x) -
    Simple power law for gamma inverse correction.
    Not used by default.

simple_gamma_correct (x) -
    Simple power law for gamma correction.
    Not used by default.

srgb_gamma_invert (x) -
    sRGB standard for gamma inverse correction.
    This is used by default.

srgb_gamma_correct (x) -
    sRGB standard for gamma correction.
    This is used by default.

Color clipping:

clip_rgb_color (rgb_color) -
    Convert a linear rgb color (nominal range 0.0 - 1.0), into a displayable
    irgb color with values in the range (0 - 255), clipping as necessary.

    The return value is a tuple, the first element is the clipped irgb color,
    and the second element is a tuple indicating which (if any) clipping processes were used.

Initialization functions:

init (
    phosphor_red   = SRGB_Red,
    phosphor_green = SRGB_Green,
    phosphor_blue  = SRGB_Blue,
    white_point    = SRGB_White) -

    Setup the conversions between CIE XYZ and linear RGB spaces.
    Also do other initializations (gamma, conversions with Luv and Lab spaces, clipping model).
    The default arguments correspond to the sRGB standard RGB space.
    The conversion is defined by supplying the chromaticities of each of
    the monitor phosphors, as well as the resulting white color when all
    of the phosphors are at full strength.
    See [Foley/Van Dam, p.587, eqn 13.27, 13.29] and [Hall, p. 239].

init_Luv_Lab_white_point (white_point) -
    Specify the white point to use for Luv/Lab conversions.

init_gamma_correction (
    display_from_linear_function = srgb_gamma_invert,
    linear_from_display_function = srgb_gamma_correct,
    gamma = STANDARD_GAMMA) -

    Setup gamma correction.
    The functions used for gamma correction/inversion can be specified,
    as well as a gamma value.
    The specified display_from_linear_function should convert a
    linear (rgb) component [proportional to light intensity] into
    displayable component [proportional to palette values].
    The specified linear_from_display_function should convert a
    displayable (rgb) component [proportional to palette values]
    into a linear component [proportional to light intensity].
    The choices for the functions:
    display_from_linear_function -
        srgb_gamma_invert [default] - sRGB standard
        simple_gamma_invert - simple power function, can specify gamma.
    linear_from_display_function -
        srgb_gamma_correct [default] - sRGB standard
        simple_gamma_correct - simple power function, can specify gamma.
    The gamma parameter is only used for the simple() functions,
    as sRGB implies an effective gamma of 2.2.

init_clipping (clip_method = CLIP_ADD_WHITE) -
    Specify the color clipping method.

References:

Foley, van Dam, Feiner and Hughes. Computer Graphics: Principles and Practice, 2nd edition,
    Addison Wesley Systems Programming Series, 1990. ISBN 0-201-12110-7.

Roy Hall, Illumination and Color in Computer Generated Imagery. Monographs in Visual Communication,
    Springer-Verlag, New York, 1989. ISBN 0-387-96774-5.

Wyszecki and Stiles, Color Science: Concepts and Methods, Quantitative Data and Formulae, 2nd edition,
    John Wiley, 1982. Wiley Classics Library Edition 2000. ISBN 0-471-39918-3.

Judd and Wyszecki, Color in Business, Science and Industry, 1975.

Kasson and Plouffe, An Analysis of Selected Computer Interchange Color Spaces,
    ACM Transactions on Graphics, Vol. 11, No. 4, October 1992.

Charles Poynton - Frequently asked questions about Gamma and Color,
    posted to comp.graphics.algorithms, 25 Jan 1995.

sRGB - http://www.color.org/sRGB.xalter - (accessed 15 Sep 2008)
    A Standard Default Color Space for the Internet: sRGB,
    Michael Stokes (Hewlett-Packard), Matthew Anderson (Microsoft), Srinivasan Chandrasekar (Microsoft),
    Ricardo Motta (Hewlett-Packard), Version 1.10, November 5, 1996.

License:

Copyright (C) 2008 Mark Kness

Author - Mark Kness - mkness@alumni.utexas.net

This file is part of ColorPy.

ColorPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

ColorPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with ColorPy.  If not, see <http://www.gnu.org/licenses/>.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy

# The xyz constructors have some special versions to handle some common situations.

def xyz_color (x, y, z = None):
    '''Construct an xyz color.  If z is omitted, set it so that x+y+z = 1.0.'''
    if z == None:
        # choose z so that x+y+z = 1.0
        z = 1.0 - (x + y)
    rtn = numpy.array ([x, y, z])
    return rtn

def xyz_normalize (xyz):
    '''Scale so that all values add to 1.0.
    This both modifies the passed argument and returns the normalized result.'''
    sum_xyz = xyz[0] + xyz[1] + xyz[2]
    if sum_xyz != 0.0:
        scale = 1.0 / sum_xyz
        xyz [0] *= scale
        xyz [1] *= scale
        xyz [2] *= scale
    return xyz

def xyz_normalize_Y1 (xyz):
    '''Scale so that the y component is 1.0.
    This both modifies the passed argument and returns the normalized result.'''
    if xyz [1] != 0.0:
        scale = 1.0 / xyz [1]
        xyz [0] *= scale
        xyz [1] *= scale
        xyz [2] *= scale
    return xyz

def xyz_color_from_xyY (x, y, Y):
    '''Given the 'little' x,y chromaticity, and the intensity Y,
    construct an xyz color.  See Foley/Van Dam p. 581, eq. 13.21.'''
    return xyz_color (
        (x/y)* Y,
        Y,
        (1.0-x-y)/(y) * Y)

# Simple constructors for the remaining models.

def rgb_color (r, g, b):
    '''Construct a linear rgb color from components.'''
    rtn = numpy.array ([r, g, b])
    return rtn

def irgb_color (ir, ig, ib):
    '''Construct a displayable integer irgb color from components.'''
    rtn = numpy.array ([ir, ig, ib], int)
    return rtn

def luv_color (L, u, v):
    '''Construct a Luv color from components.'''
    rtn = numpy.array ([L, u, v])
    return rtn

def lab_color (L, a, b):
    '''Construct a Lab color from components.'''
    rtn = numpy.array ([L, a, b])
    return rtn

#
# Definitions of some standard values for colors and conversions
#

# Chromaticities of various standard phosphors and white points.

# sRGB (ITU-R BT.709) standard phosphor chromaticities
SRGB_Red   = xyz_color (0.640, 0.330)
SRGB_Green = xyz_color (0.300, 0.600)
SRGB_Blue  = xyz_color (0.150, 0.060)
SRGB_White = xyz_color (0.3127, 0.3290)  # D65

# HDTV standard phosphors, from Poynton [Color FAQ] p. 9
#   These are claimed to be similar to typical computer monitors
HDTV_Red   = xyz_color (0.640, 0.330)
HDTV_Green = xyz_color (0.300, 0.600)
HDTV_Blue  = xyz_color (0.150, 0.060)
# use D65 as white point for HDTV

# SMPTE phosphors
#   However, Hall [p. 188] notes that TV expects values calibrated for NTSC
#   even though actual phosphors are as below.
# From Hall p. 118, and Kasson p. 400
SMPTE_Red   = xyz_color (0.630, 0.340)
SMPTE_Green = xyz_color (0.310, 0.595)
SMPTE_Blue  = xyz_color (0.155, 0.070)
# use D65 as white point for SMPTE

# NTSC phosphors [original standard for TV, but no longer used in TV sets]
# From Hall p. 119 and Foley/Van Dam p. 589
NTSC_Red   = xyz_color (0.670, 0.330)
NTSC_Green = xyz_color (0.210, 0.710)
NTSC_Blue  = xyz_color (0.140, 0.080)
# use D65 as white point for NTSC

# Typical short persistence phosphors from Foley/Van Dam p. 583
FoleyShort_Red   = xyz_color (0.61, 0.35)
FoleyShort_Green = xyz_color (0.29, 0.59)
FoleyShort_Blue  = xyz_color (0.15, 0.063)

# Typical long persistence phosphors from Foley/Van Dam p. 583
FoleyLong_Red   = xyz_color (0.62, 0.33)
FoleyLong_Green = xyz_color (0.21, 0.685)
FoleyLong_Blue  = xyz_color (0.15, 0.063)

# Typical TV phosphors from Judd/Wyszecki p. 239
Judd_Red   = xyz_color (0.68, 0.32)       # Europium Yttrium Vanadate
Judd_Green = xyz_color (0.28, 0.60)       # Zinc Cadmium Sulfide
Judd_Blue  = xyz_color (0.15, 0.07)       # Zinc Sulfide

# White points [all are for CIE 1931 for small field of view]
#   These are from Judd/Wyszecki
WhiteA   = xyz_color (0.4476, 0.4074)      # approx 2856 K
WhiteB   = xyz_color (0.3484, 0.3516)      # approx 4874 K
WhiteC   = xyz_color (0.3101, 0.3162)      # approx 6774 K
WhiteD55 = xyz_color (0.3324, 0.3475)      # approx 5500 K
WhiteD65 = xyz_color (0.3127, 0.3290)      # approx 6500 K
WhiteD75 = xyz_color (0.2990, 0.3150)      # approx 7500 K

# Blackbody white points [this empirically gave good results]
Blackbody6500K = xyz_color (0.3135, 0.3237)
Blackbody6600K = xyz_color (0.3121, 0.3223)
Blackbody6700K = xyz_color (0.3107, 0.3209)
Blackbody6800K = xyz_color (0.3092, 0.3194)
Blackbody6900K = xyz_color (0.3078, 0.3180)
Blackbody7000K = xyz_color (0.3064, 0.3166)

# MacBeth Color Checker white patch
#   Using this as white point will force MacBeth chart entry to equal machine RGB
MacBethWhite = xyz_color (0.30995, 0.31596, 0.37409)

# Also see Judd/Wyszecki p.164 for colors of Planck Blackbodies

# Some standard xyz/rgb conversion matricies, which assume particular phosphors.
# These are useful for testing.

# sRGB, from http://www.color.org/sRGB.xalter
srgb_rgb_from_xyz_matrix = numpy.array ([
    [ 3.2410, -1.5374, -0.4986],
    [-0.9692,  1.8760,  0.0416],
    [ 0.0556, -0.2040,  1.0570]
])

# SMPTE conversions, from Kasson p. 400
smpte_xyz_from_rgb_matrix = numpy.array ([
    [0.3935, 0.3653, 0.1916],
    [0.2124, 0.7011, 0.0865],
    [0.0187, 0.1119, 0.9582]
])
smpte_rgb_from_xyz_matrix = numpy.array ([
    [ 3.5064, -1.7400, -0.5441],
    [-1.0690,  1.9777,  0.0352],
    [ 0.0563, -0.1970,  1.0501]
])

#
# Conversions between CIE XYZ and RGB colors.
#     Assumptions must be made about the specific device to construct the conversions.
#

def rgb_from_xyz (xyz):
    '''Convert an xyz color to rgb.'''
    return color_converter.rgb_from_xyz (xyz)

def xyz_from_rgb (rgb):
    '''Convert an rgb color to xyz.'''
    return color_converter.xyz_from_rgb (rgb)

# Conversion from xyz to rgb, while also scaling the brightness to the maximum displayable.

def brightest_rgb_from_xyz (xyz, max_component=1.0):
    '''Convert the xyz color to rgb, and scale to maximum displayable brightness, so one of the components will be 1.0 (or max_component).'''
    rgb = rgb_from_xyz (xyz)
    max_rgb = max (rgb)
    if max_rgb != 0.0:
        scale = max_component / max_rgb
        rgb *= scale
    return rgb

#
# Color model conversions to (nearly) perceptually uniform spaces Luv and Lab.
#

# Luminance function [of Y value of an XYZ color] used in Luv and Lab. See [Kasson p.399] for details.
# The linear range coefficient L_LUM_C has more digits than in the paper,
# this makes the function more continuous over the boundary.

L_LUM_A      = 116.0
L_LUM_B      = 16.0
L_LUM_C      = 903.29629551307664
L_LUM_CUTOFF = 0.008856

def L_luminance (y):
    '''L coefficient for Luv and Lab models.'''
    if y > L_LUM_CUTOFF:
        return L_LUM_A * math.pow (y, 1.0/3.0) - L_LUM_B
    else:
        # linear range
        return L_LUM_C * y

def L_luminance_inverse (L):
    '''Inverse of L_luminance().'''
    if L <= (L_LUM_C * L_LUM_CUTOFF):
        # linear range
        y = L / L_LUM_C
    else:
        t = (L + L_LUM_B) / L_LUM_A
        y = math.pow (t, 3)
    return y

# Utility function for Luv

def uv_primes (xyz):
    '''Luv utility.'''
    x = xyz [0]
    y = xyz [1]
    z = xyz [2]
    w_denom = x + 15.0 * y + 3.0 * z
    if w_denom != 0.0:
        u_prime = 4.0 * x / w_denom
        v_prime = 9.0 * y / w_denom
    else:
        # this should only happen when x=y=z=0 [i.e. black] since xyz values are positive
        u_prime = 0.0
        v_prime = 0.0
    return (u_prime, v_prime)

def uv_primes_inverse (u_prime, v_prime, y):
    '''Inverse of uv_primes(). We will always have y known when this is called.'''
    if v_prime != 0.0:
        # normal
        w_denom = (9.0 * y) / v_prime
        x = 0.25 * u_prime * w_denom
        y = y
        z = (w_denom - x - 15.0 * y) / 3.0
    else:
        # should only happen when color is totally black
        x = 0.0
        y = 0.0
        z = 0.0
    xyz = xyz_color (x, y, z)
    return xyz

# Utility function for Lab
#     See [Kasson p.399] for details.
#     The linear range coefficient has more digits than in the paper,
#     this makes the function more continuous over the boundary.

LAB_F_A = 7.7870370302851422
LAB_F_B = (16.0/116.0)
# same cutoff as L_luminance()

def Lab_f (t):
    '''Lab utility function.'''
    if t > L_LUM_CUTOFF:
        return math.pow (t, 1.0/3.0)
    else:
        # linear range
        return LAB_F_A * t + LAB_F_B

def Lab_f_inverse (F):
    '''Inverse of Lab_f().'''
    if F <= (LAB_F_A * L_LUM_CUTOFF + LAB_F_B):
        # linear range
        t = (F - LAB_F_B) / LAB_F_A
    else:
        t = math.pow (F, 3)
    return t

# Conversions between standard device independent color space (CIE XYZ)
# and the almost perceptually uniform space Luv.

def luv_from_xyz (xyz):
    '''Convert CIE XYZ to Luv.'''
    return color_converter.luv_from_xyz(xyz)

def xyz_from_luv (luv):
    '''Convert Luv to CIE XYZ.  Inverse of luv_from_xyz().'''
    return color_converter.xyz_from_luv(luv)

# Conversions between standard device independent color space (CIE XYZ)
# and the almost perceptually uniform space Lab.

def lab_from_xyz (xyz):
    '''Convert color from CIE XYZ to Lab.'''
    return color_converter.lab_from_xyz(xyz)

def xyz_from_lab (Lab):
    '''Convert color from Lab to CIE XYZ.  Inverse of lab_from_xyz().'''
    return color_converter.xyz_from_lab(Lab)

# Gamma correction
#
# Non-gamma corrected rgb values, also called non-linear rgb values,
# correspond to palette register entries [although here they are kept
# in the range 0.0 to 1.0.]  The numerical values are not proportional
# to the amount of light energy present.
#
# Gamma corrected rgb values, also called linear rgb values,
# do not correspond to palette entries.  The numerical values are
# proportional to the amount of light energy present.
#
# This effect is particularly significant with CRT displays.
# With LCD displays, it is less clear (at least to me), what the genuinely
# correct correction should be.

# Available gamma correction methods.
GAMMA_CORRECT_POWER = 0    # Simple power law, using supplied gamma exponent.
GAMMA_CORRECT_SRGB  = 1    # sRGB correction formula.

# sRGB standard effective gamma.  This exponent is not applied explicitly.
STANDARD_GAMMA = 2.2

# Although NTSC specifies a gamma of 2.2 as standard, this is designed
# to account for the dim viewing environments typical of TV, but not
# computers.  Well-adjusted CRT displays have a true gamma in the range
# 2.35 through 2.55.  We use the physical gamma value here, not 2.2,
# thus not correcting for a dim viewing environment.
# [Poynton, Gamma FAQ p.5, p.9, Hall, p. 121]
POYNTON_GAMMA = 2.45

# Simple power laws for gamma correction

def simple_gamma_invert (x, gamma_exponent):
    '''Simple power law for gamma inverse correction.'''
    if x <= 0.0:
        return x
    else:
        return math.pow (x, 1.0 / gamma_exponent)

def simple_gamma_correct (x, gamma_exponent):
    '''Simple power law for gamma correction.'''
    if x <= 0.0:
        return x
    else:
        return math.pow (x, gamma_exponent)

# sRGB gamma correction - http://www.color.org/sRGB.xalter
# The effect of the equations is to closely fit a straightforward
# gamma 2.2 curve with an slight offset to allow for invertability in
# integer math. Therefore, we are maintaining consistency with the
# gamma 2.2 legacy images and the video industry.

def srgb_gamma_invert (x):
    '''sRGB standard for gamma inverse correction.'''
    if x <= 0.00304:
        rtn = 12.92 * x
    else:
        rtn = 1.055 * math.pow (x, 1.0/2.4) - 0.055
    return rtn

def srgb_gamma_correct (x):
    '''sRGB standard for gamma correction.'''
    if x <= 0.03928:
        rtn = x / 12.92
    else:
        rtn = math.pow ((x + 0.055) / 1.055, 2.4)
    return rtn

#
# New gamma correction...
#

class GammaCorrect(object):
    ''' Gamma correction formulas as used in several standards.

    'display' - Color values as would be used in display code.
    'linear'  - Color values with numbers proportional to physical intensity.
    Both are nominally in the range 0.0 - 1.0.

    The curves have a linear region near black,
    and approximately exponential for visibly bright colors.
    The linear region avoids numerical trouble near zero.

    Note that the effective gamma exponent that this model provides,
    is not exactly the same value as the gamma that is nominally supplied.
    For example, sRGB uses the number gamma=2.4, but its curve actually
    better approximates an exponent of 2.2.

    C_display = Phi * C_linear,                    C_linear <  K0 / Phi
    C_display = (1+a) * C_linear^(1/gamma) - a,    C_linear >= K0 / Phi

    C_linear = C_display / Phi,                    C_display <  K0
    C_linear = ((C_display + a) / (1+a))^gamma,    C_display >= K0

    The two regions (linear/exponential) ought to connect sensibly.
    '''

    def __init__(self,
        gamma,    # gamma exponent.
        a,        # offset.
        K0,       # intensity cutoff.
        Phi):     # linear scaling.
        self.gamma = float(gamma)
        self.a     = float(a)
        self.K0    = float(K0)
        self.Phi   = float(Phi)
        # Precompute.
        self.one_plus_a  = 1.0 + self.a
        self.inv_gamma   = 1.0 / self.gamma
        self.K0_over_Phi = self.K0 / self.Phi
        # Enforce continuity at the 'edge of black'.
        # This discards the original K0 and Phi.
        self.set_continuous_slope()
        # Improve K0 values.
        # This does nothing after set_continuous_slope().
        # Before it does not seem to converge!
        for i in range(4):
            #self.improve_K0()
            self.improve_Phi()

    def display_from_linear(self, C_linear):
        ''' Convert physical intensity to display values. '''
        if C_linear < self.K0_over_Phi:
            # Linear region.
            C_display = self.Phi * C_linear
        else:
            # Pseudo-exponential region.
            C_linear_inv_gamma = math.pow(C_linear, self.inv_gamma)
            C_display = self.one_plus_a * C_linear_inv_gamma - self.a
        return C_display

    def linear_from_display(self, C_display):
        ''' Convert display values to physical intensity. '''
        if C_display < self.K0:
            # Linear region.
            C_linear = C_display / self.Phi
        else:
            # Pseudo-exponential region.
            C_display_term = (C_display + self.a) / self.one_plus_a
            C_linear = math.pow(C_display_term, self.gamma)
        return C_linear

    # Continuity between the linear and pseudo-exponential regions requires:
    #     ((K0 + a) / (1+a))^gamma = K0 / Phi
    #
    # This can be used to check a K0 estimate.
    # At present it does not seem to be a useful iteration.
    # Or you could calculate Phi.
    # That works sometimes but not reliably.

    def improve_K0(self):
        ''' Check K0 value. This converges poorly as an improvement attempt. '''
        K0_start = self.K0
        lhs_term = ((self.K0 + self.a) / (self.one_plus_a))
        lhs = math.pow(lhs_term, self.gamma)
        rhs = self.K0 / self.Phi
        # New K0 value that was hoped better, but actually seems worse!
        K0_better = self.Phi * lhs
        self.K0 = K0_better
        msg = 'K0_start=%g    lhs=%g    rhs=%g    K0_better=%g' % (
            K0_start, lhs, rhs, K0_better)
        print (msg)

    def improve_Phi(self):
        ''' Automatically set Phi to enforce continuity at the edge of black. '''
        # This seems to work well for sRGB and poorly for UHDTV.
        # Perhaps there are two solutions???
        Phi_start = self.Phi
        lhs_term = ((self.K0 + self.a) / (self.one_plus_a))
        lhs = math.pow(lhs_term, self.gamma)
        rhs = self.K0 / self.Phi
        # New Phi value that is ideally better.
        # Sometimes yes, sometimes no.
        Phi_better = self.K0 / lhs
        self.Phi = Phi_better
        self.K0_over_Phi = self.K0 / self.Phi
        msg = 'Phi_start=%.12f    lhs=%g    rhs=%g    Phi_better=%.12f' % (
            Phi_start, lhs, rhs, Phi_better)
        print (msg)

    # Continuity of value and slope requires:
    #     K0 = a / (gamma - 1)
    #     Phi = ((1+a)^gamma * (gamma-1)^(gamma-1)) /
    #           (a^(gamma-1) * gamma^gamma)
    #
    # This seems to make a lot of sense to enforce.
    # The K0 and Phi values apply to the 'edge of black' and so
    # it is unlikely they are really carefully chosen, while the
    # continuity condition seems natural. And it also makes sense to
    # enforce the 'edge of black' all at once.

    def set_continuous_slope(self):
        ''' Automatically set K0 and Phi to enforce slope continuity. '''
        K0_start  = self.K0
        Phi_start = self.Phi
        K0_better = (self.a / (self.gamma - 1.0))
        Phi_term_1 = math.pow(self.one_plus_a, self.gamma)
        Phi_term_2 = math.pow(self.gamma - 1.0, self.gamma - 1.0)
        Phi_term_3 = math.pow(self.a, self.gamma - 1.0)
        Phi_term_4 = math.pow(self.gamma, self.gamma)
        Phi_better = (Phi_term_1 * Phi_term_2) / (Phi_term_3 * Phi_term_4)
        self.K0  = K0_better
        self.Phi = Phi_better
        self.K0_over_Phi = self.K0 / self.Phi
        msg = 'K0_start=%g    Phi_start=%g    K0_better=%.12f    Phi_better=%.12f' % (
            K0_start, Phi_start, K0_better, Phi_better)
        print (msg)

# sRGB gamma correction, for HDTV.
#   http://en.wikipedia.org/wiki/SRGB, accessed 1 Apr 2015
# Note that, despite the nominal gamma=2.4, the function overall is designed
# to approximate gamma=2.2.

srgb_gamma_corrector = GammaCorrect(
    gamma=2.4, a=0.055, K0=0.03928, Phi=12.92)

# Rec 2020 gamma correction, for UHDTV.
#   https://en.wikipedia.org/wiki/Rec._2020, accessed 1 Apr 2015.

# Rec 2020/UHDTV for 10 bits per component.
uhdtv_10_gamma_corrector = GammaCorrect(
	gamma=(1.0/0.45), a=0.099, K0=0.01, Phi=4.5)	# FIXME: K0 is wrong.

# Rec 2020/UHDTV for 12 bits per component.
uhdtv_12_gamma_corrector = GammaCorrect(
	gamma=(1.0/0.45), a=0.0993, K0=0.01, Phi=4.5)	# FIXME: K0 is wrong.

#
# Color clipping - Physical color values may exceed the what the display can show,
#   either because the color is too pure (indicated by negative rgb values), or
#   because the color is too bright (indicated by rgb values > 1.0).
#   These must be clipped to something displayable.
#

# possible color clipping methods
CLIP_CLAMP_TO_ZERO = 0
CLIP_ADD_WHITE     = 1

def clip_rgb_color (rgb_color):
    '''Convert a linear rgb color (nominal range 0.0 - 1.0), into a displayable
    irgb color with values in the range (0 - 255), clipping as necessary.

    The return value is a tuple, the first element is the clipped irgb color,
    and the second element is a tuple indicating which (if any) clipping processes were used.
    '''
    return color_converter.clip_rgb_color (rgb_color)

#
# Conversions between linear rgb colors (range 0.0 - 1.0, values proportional to light intensity)
# and displayable irgb colors (range 0 - 255, values corresponding to hardware palette values).
#
# Displayable irgb colors can be represented as hex strings, like '#AB05B4'.
#

def irgb_string_from_irgb (irgb):
    '''Convert a displayable irgb color (0-255) into a hex string.'''
    # ensure that values are in the range 0-255
    for index in range (0, 3):
        irgb [index] = min (255, max (0, irgb [index]))
    # convert to hex string
    irgb_string = '#%02X%02X%02X' % (irgb [0], irgb [1], irgb [2])
    return irgb_string

def irgb_from_irgb_string (irgb_string):
    '''Convert a color hex string (like '#AB13D2') into a displayable irgb color.'''
    strlen = len (irgb_string)
    if strlen != 7:
        raise ValueError('irgb_string_from_irgb(): Expecting 7 character string like #AB13D2')
    if irgb_string [0] != '#':
        raise ValueError('irgb_string_from_irgb(): Expecting 7 character string like #AB13D2')
    irs = irgb_string [1:3]
    igs = irgb_string [3:5]
    ibs = irgb_string [5:7]
    ir = int (irs, 16)
    ig = int (igs, 16)
    ib = int (ibs, 16)
    irgb = irgb_color (ir, ig, ib)
    return irgb

def irgb_from_rgb (rgb):
    '''Convert a (linear) rgb value (range 0.0 - 1.0) into a 0-255 displayable integer irgb value (range 0 - 255).'''
    return color_converter.irgb_from_rgb (rgb)

def rgb_from_irgb (irgb):
    '''Convert a displayable (gamma corrected) irgb value (range 0 - 255) into a linear rgb value (range 0.0 - 1.0).'''
    return color_converter.rgb_from_irgb (irgb)

def irgb_string_from_rgb (rgb):
    '''Clip the rgb color, convert to a displayable color, and convert to a hex string.'''
    return irgb_string_from_irgb (irgb_from_rgb (rgb))

# Multi-level conversions, for convenience

def irgb_from_xyz (xyz):
    '''Convert an xyz color directly into a displayable irgb color.'''
    return irgb_from_rgb (rgb_from_xyz (xyz))

def irgb_string_from_xyz (xyz):
    '''Convert an xyz color directly into a displayable irgb color hex string.'''
    return irgb_string_from_rgb (rgb_from_xyz (xyz))

#
# Object to hold color conversion values.
#

# Note: In the previous version of this code, you could specify arbitrary
# functions for the gamma conversions. This is now not directly possible,
# instead you choose the method from the enumerated list. This should be
# easier and more reliable for most all actual usage. If the arbitary
# functions are needed, a new gamma method is needed.
#
# FIXME: Add this new gamma method.
# FIXME: Should be able to specify maximum value rather than bit depth.

class ColorConverter(object):
    ''' An object to keep track of how to convert between color spaces. '''

    def __init__ (self,
        phosphor_red   = SRGB_Red,
        phosphor_green = SRGB_Green,
        phosphor_blue  = SRGB_Blue,
        white_point    = SRGB_White,
        gamma_method   = GAMMA_CORRECT_SRGB,
        gamma_value    = STANDARD_GAMMA,
        clip_method    = CLIP_ADD_WHITE,
        bit_depth      = 8):
        ''' Initialize the color conversions. '''
        # xyz <-> rgb conversions need phosphor chromaticities and white point.
        self.init_rgb_xyz(
            phosphor_red, phosphor_green, phosphor_blue, white_point)
        # xyz <-> Luv and Lab conversions need white point.
        self.init_Luv_Lab_white_point(white_point)
        # Gamma correction method.
        self.init_gamma_correction(gamma_method, gamma_value)
        # Clipping method.
        self.init_clipping(clip_method)
        # Bit depth for integer rgb values.
        self.init_bit_depth(bit_depth)

    def init_rgb_xyz(self,
        phosphor_red,
        phosphor_green,
        phosphor_blue,
        white_point):
        '''Setup the conversions between CIE XYZ and linear RGB spaces.

        The default arguments correspond to the sRGB standard RGB space.
        The conversion is defined by supplying the chromaticities of each of
        the monitor phosphors, as well as the resulting white color when all
        of the phosphors are at full strength.

        See [Foley/Van Dam, p.587, eqn 13.27, 13.29] and [Hall, p. 239].
        '''
        # xyz colors of the monitor phosphors (and full white).
        self.PhosphorRed   = phosphor_red
        self.PhosphorGreen = phosphor_green
        self.PhosphorBlue  = phosphor_blue
        self.PhosphorWhite = white_point
        phosphor_matrix = numpy.column_stack ((phosphor_red, phosphor_green, phosphor_blue))
        # Normalize white point to Y=1.0.
        normalized_white = white_point.copy()
        xyz_normalize_Y1 (normalized_white)
        # Determine intensities of each phosphor by solving:
        #     phosphor_matrix * intensity_vector = white_point
        intensities = numpy.linalg.solve (phosphor_matrix, normalized_white)
        # Construct xyz_from_rgb matrix from the results.
        self.xyz_from_rgb_matrix = numpy.column_stack (
            (phosphor_red   * intensities [0],
             phosphor_green * intensities [1],
             phosphor_blue  * intensities [2]))
        # Invert to get rgb_from_xyz matrix.
        self.rgb_from_xyz_matrix = numpy.linalg.inv (self.xyz_from_rgb_matrix)

    def init_Luv_Lab_white_point(self, white_point):
        ''' Specify the white point to use for Luv/Lab conversions. '''
        self.reference_white = white_point.copy()
        xyz_normalize_Y1 (self.reference_white)
        self.reference_u_prime, self.reference_v_prime = uv_primes (self.reference_white)

    def init_gamma_correction(self, gamma_method, gamma_value):
        '''Specify the gamma correction method.

        Gamma correction converts rgb components, in 0.0 - 1.0 range, between
        linear values, proportional to light intensity, and
        displayable values, proportional to palette values.

        The choices for the method:
        GAMMA_CORRECT_POWER:
            Apply a simple exponent conversion.
            The gamma value should be specified.
        GAMMA_CORRECT_SRGB:
            Apply the sRGB correction formula.
            The gamma exponent is ignored. It is effectively 2.2.
        '''
        if not gamma_method in [GAMMA_CORRECT_POWER, GAMMA_CORRECT_SRGB]:
            raise ValueError('Invalid gamma correction method %s' % (str(gamma_method)))
        self.gamma_method = gamma_method
        self.gamma_value  = gamma_value

    def init_clipping(self, clip_method):
        '''Specify the color clipping method.'''
        if not clip_method in [CLIP_CLAMP_TO_ZERO, CLIP_ADD_WHITE]:
            raise ValueError('Invalid color clipping method %s' % (str(clip_method)))
        self.clip_method = clip_method

    def init_bit_depth(self, bit_depth):
        ''' Initialize the bit depth for displayable integer rgb colors. '''
        self.bit_depth = bit_depth
        self.max_value = (1 << self.bit_depth) - 1

    def dump(self):
        ''' Print some info about the color conversions. '''
        print ('xyz_from_rgb', str (self.xyz_from_rgb_matrix))
        print ('rgb_from_xyz', str (self.rgb_from_xyz_matrix))
        # Bit depth.
        print ('bit_depth = %d' % (self.bit_depth))
        print ('max_value = %d' % (self.max_value))

    # Conversions between xyz and rgb.
    # (rgb here is linear, not gamma adjusted.)

    def rgb_from_xyz(self, xyz):
        '''Convert an xyz color to rgb.'''
        return numpy.dot (self.rgb_from_xyz_matrix, xyz)

    def xyz_from_rgb(self, rgb):
        '''Convert an rgb color to xyz.'''
        return numpy.dot (self.xyz_from_rgb_matrix, rgb)

    # Conversions between standard device independent color space (CIE XYZ)
    # and the almost perceptually uniform space Luv.

    def luv_from_xyz(self, xyz):
        '''Convert CIE XYZ to Luv.'''
        y = xyz [1]
        y_p = y / self.reference_white [1]       # reference_white [1] is probably always 1.0.
        u_prime, v_prime = uv_primes (xyz)
        L = L_luminance (y_p)
        u = 13.0 * L * (u_prime - self.reference_u_prime)
        v = 13.0 * L * (v_prime - self.reference_v_prime)
        luv = luv_color (L, u, v)
        return luv

    def xyz_from_luv(self, luv):
        '''Convert Luv to CIE XYZ.  Inverse of luv_from_xyz().'''
        L = luv [0]
        u = luv [1]
        v = luv [2]
        # Invert L_luminance() to get y.
        y = L_luminance_inverse (L)
        if L != 0.0:
            # Color is not totally black.
            # Get u_prime, v_prime.
            L13 = 13.0 * L
            u_prime = self.reference_u_prime + (u / L13)
            v_prime = self.reference_v_prime + (v / L13)
            # Get xyz color.
            xyz = uv_primes_inverse (u_prime, v_prime, y)
        else:
            # Color is black.
            xyz = xyz_color (0.0, 0.0, 0.0)
        return xyz

    # Conversions between standard device independent color space (CIE XYZ)
    # and the almost perceptually uniform space Lab.

    def lab_from_xyz(self, xyz):
        '''Convert color from CIE XYZ to Lab.'''
        x = xyz [0]
        y = xyz [1]
        z = xyz [2]

        x_p = x / self.reference_white [0]
        y_p = y / self.reference_white [1]
        z_p = z / self.reference_white [2]

        f_x = Lab_f (x_p)
        f_y = Lab_f (y_p)
        f_z = Lab_f (z_p)

        L = L_luminance (y_p)
        a = 500.0 * (f_x - f_y)
        b = 200.0 * (f_y - f_z)
        Lab = lab_color (L, a, b)
        return Lab

    def xyz_from_lab(self, Lab):
        '''Convert color from Lab to CIE XYZ.  Inverse of lab_from_xyz().'''
        L = Lab [0]
        a = Lab [1]
        b = Lab [2]
        # invert L_luminance() to get y_p
        y_p = L_luminance_inverse (L)
        # calculate f_y
        f_y = Lab_f (y_p)
        # solve for f_x and f_z
        f_x = f_y + (a / 500.0)
        f_z = f_y - (b / 200.0)
        # invert Lab_f() to get x_p and z_p
        x_p = Lab_f_inverse (f_x)
        z_p = Lab_f_inverse (f_z)
        # multiply by reference white to get xyz
        x = x_p * self.reference_white [0]
        y = y_p * self.reference_white [1]
        z = z_p * self.reference_white [2]
        xyz = xyz_color (x, y, z)
        return xyz

    # Conversion of linear rgb color (range 0.0 - 1.0) to displayable values (range 0 - 255).

    # Clipping of undisplayable colors.

    def clip_color_clamp(self, rgb):
        ''' Clip an rgb color to remove negative components.
        Any negative components are zeroed. '''
        # The input color is modified as necessary.
        clipped = False
        # Set negative rgb values to zero.
        if rgb [0] < 0.0:
            rgb [0] = 0.0
            clipped = True
        if rgb [1] < 0.0:
            rgb [1] = 0.0
            clipped = True
        if rgb [2] < 0.0:
            rgb [2] = 0.0
            clipped = True
        return clipped

    def clip_color_whiten(self, rgb):
        ''' Clip an rgb color to remove negative components.
        White is added as necessary to remove any negative components. '''
        # The input color is modified as necessary.
        clipped = False
        # Add enough white to make all rgb values nonnegative.
        rgb_min = min (0.0, min (rgb))
        # Get scaling factor to maintain max rgb after adding white.
        rgb_max = max (rgb)
        scaling = 1.0
        if rgb_max > 0.0:
            scaling = rgb_max / (rgb_max - rgb_min)
        # Add white and scale.
        if rgb_min < 0.0:
            rgb [0] = scaling * (rgb [0] - rgb_min);
            rgb [1] = scaling * (rgb [1] - rgb_min);
            rgb [2] = scaling * (rgb [2] - rgb_min);
            clipped = True
        return clipped

    def clip_color_intensity(self, rgb):
        ''' Scale an rgb color if needed to the component range 0.0 - 1.0. '''
        # The input color is modified as necessary.
        clipped = False
        rgb_max = max (rgb)
        # Does not actually overflow until 2^B * intensity > (2^B + 0.5).
        intensity_cutoff = 1.0 + (0.5 / self.max_value)
        if rgb_max > intensity_cutoff:
            scaling = intensity_cutoff / rgb_max
            rgb *= scaling
            clipped = True
        return clipped

    # Gamma correction, to convert between linear and displayable values.
    # Linear = Component value is proportional to physical light intensity.
    # Displayable = Component value is appropriate to display on monitor.

    def gamma_display_from_linear_component(self, x):
        ''' Gamma adjust an rgb component (range 0.0 - 1.0) to convert
        from linear to displayable values. '''
        # This is gamma inversion, not gamma correction.
        if self.gamma_method == GAMMA_CORRECT_POWER:
            y = simple_gamma_invert (x, self.gamma_value)
        elif self.gamma_method == GAMMA_CORRECT_SRGB:
            y = srgb_gamma_invert (x)
        else:
            raise ValueError('Invalid gamma correction method %s' % (str(self.gamma_method)))
        return y

    def gamma_linear_from_display_component(self, x):
        ''' Gamma adjust an rgb component (range 0.0 - 1.0) to convert
        from displayable to linear values. '''
        # This is gamma correction.
        if self.gamma_method == GAMMA_CORRECT_POWER:
            y = simple_gamma_correct (x, self.gamma_value)
        elif self.gamma_method == GAMMA_CORRECT_SRGB:
            y = srgb_gamma_correct (x)
        else:
            raise ValueError('Invalid gamma correction method %s' % (str(self.gamma_method)))
        return y

    def gamma_display_from_linear(self, rgb):
        ''' Gamma adjust an rgb color (range 0.0 - 1.0) to convert
        from linear to displayable values. '''
        rgb[0] = self.gamma_display_from_linear_component(rgb[0])
        rgb[1] = self.gamma_display_from_linear_component(rgb[1])
        rgb[2] = self.gamma_display_from_linear_component(rgb[2])

    def gamma_linear_from_display(self, rgb):
        ''' Gamma adjust an rgb color (range 0.0 - 1.0) to convert
        from displayable to linear values. '''
        rgb[0] = self.gamma_linear_from_display_component(rgb[0])
        rgb[1] = self.gamma_linear_from_display_component(rgb[1])
        rgb[2] = self.gamma_linear_from_display_component(rgb[2])

    # Scaling from 0.0 - 1.0 range to integer values 0 - 2^(bitdepth) - 1.

    def scale_int_from_float(self, rgb):
        ''' Scale a color with component range 0.0 - 1.0 to integer values
        in range 0 - 2^(bitdepth) - 1. '''
        ir = round (self.max_value * rgb [0])
        ig = round (self.max_value * rgb [1])
        ib = round (self.max_value * rgb [2])
        # Ensure that values are in the valid range.
        # This is redundant if the value was properly clipped, but make sure.
        ir = min (self.max_value, max (0, ir))
        ig = min (self.max_value, max (0, ig))
        ib = min (self.max_value, max (0, ib))
        irgb = irgb_color (ir, ig, ib)
        return irgb

    def scale_float_from_int(self, irgb):
        ''' Scale a color with integer components 0 - 2^(bitdepth) - 1
        to floating point values in range 0.0 - 1.0. '''
        # Scale to 0.0 - 1.0.
        r = float (irgb [0]) / self.max_value
        g = float (irgb [1]) / self.max_value
        b = float (irgb [2]) / self.max_value
        rgb = rgb_color (r, g, b)
        return rgb

    def clip_rgb_color(self, rgb_color):
        '''Convert a linear rgb color (nominal range 0.0 - 1.0), into a displayable
        irgb color with values in the range (0 - 255), clipping as necessary.

        The return value is a tuple, the first element is the clipped irgb color,
        and the second element is a tuple indicating which (if any) clipping processes were used.
        '''
        clipped_chromaticity = False
        clipped_intensity = False

        rgb = rgb_color.copy()

        # clip chromaticity if needed (negative rgb values)
        if self.clip_method == CLIP_CLAMP_TO_ZERO:
            clipped_chromaticity = self.clip_color_clamp(rgb)
        elif self.clip_method == CLIP_ADD_WHITE:
            clipped_chromaticity = self.clip_color_whiten(rgb)
        else:
            raise ValueError('Invalid color clipping method %s' % (str(self.clip_method)))

        # clip intensity if needed (rgb values > 1.0) by scaling
        clipped_intensity = self.clip_color_intensity(rgb)

        # gamma correction
        self.gamma_display_from_linear(rgb)

        # Scale to 0 - 2^B - 1.
        irgb = self.scale_int_from_float(rgb)
        return (irgb, (clipped_chromaticity, clipped_intensity))

    # Conversions between linear rgb colors (0.0 - 1.0 range) and
    # displayable irgb values (0 - 2^B - 1 range).

    def irgb_from_rgb(self, rgb):
        '''Convert a (linear) rgb value (range 0.0 - 1.0) into a displayable integer irgb value (range 0 - 2^B - 1).'''
        result = self.clip_rgb_color (rgb)
        (irgb, (clipped_chrom,clipped_int)) = result
        return irgb

    def rgb_from_irgb(self, irgb):
        '''Convert a displayable (gamma corrected) irgb value (range 0 - 2^B - 1) into a linear rgb value (range 0.0 - 1.0).'''
        # Scale to 0.0 - 1.0, and gamma correct.
        rgb = self.scale_float_from_int(irgb)
        self.gamma_linear_from_display(rgb)
        return rgb

#
# Initialization - Initialize to sRGB at module startup.
#   If a different rgb model is needed, then the startup can be re-done to set the new conditions.
#

color_converter = None

def init (
    phosphor_red   = SRGB_Red,
    phosphor_green = SRGB_Green,
    phosphor_blue  = SRGB_Blue,
    white_point    = SRGB_White,
    gamma_method   = GAMMA_CORRECT_SRGB,
    gamma_value    = STANDARD_GAMMA,
    clip_method    = CLIP_ADD_WHITE,
    bit_depth      = 8):
    ''' Initialize. '''
    global color_converter
    color_converter = ColorConverter(
        phosphor_red   = phosphor_red,
        phosphor_green = phosphor_green,
        phosphor_blue  = phosphor_blue,
        white_point    = white_point,
        gamma_method   = gamma_method,
        gamma_value    = gamma_value,
        clip_method    = clip_method,
        bit_depth      = bit_depth)
    #color_converter.dump()


init()
# Default conversions setup on module load
