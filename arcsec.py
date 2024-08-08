from functions import crossmatch_catalogs, add_offsets, plot_offset
from numpy import loadtxt
import os
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import sys
import numpy as np


# Change directory to the location of the FITS file
os.chdir('/home/apatrick/Code')  # Adjust this path if necessary


_