from functions import crossmatch_catalogs, add_offsets, plot_offset
from numpy import loadtxt
import os
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import sys
import matplotlib.pyplot as plt
import numpy as np


# Change directory to the location of the FITS file
os.chdir('/home/apatrick/Code')  # Adjust this path if necessary


cube = 28
HST = 814

# Define the paths to the FITS files
arc_1 = f'outputs/matched_catalog_{cube}_{HST}_1.0.fits'
arc_15 = f'outputs/matched_catalog_{cube}_{HST}_1.5.fits'
arc_2 = f'outputs/matched_catalog_{cube}_{HST}_2.0.fits'
arc_25 = f'outputs/matched_catalog_{cube}_{HST}_2.5.fits'
arc_3 = f'outputs/matched_catalog_{cube}_{HST}_3.0.fits'
arc_4 = f'outputs/matched_catalog_{cube}_{HST}_4.0.fits'
arc_5 = f'outputs/matched_catalog_{cube}_{HST}_5.0.fits'
arc_6 = f'outputs/matched_catalog_{cube}_{HST}_6.0.fits'

# Load the catalogs from FITS files
arc_1 = fits.open(arc_1)[1].data
arc_15 = fits.open(arc_15)[1].data
arc_2 = fits.open(arc_2)[1].data
arc_25 = fits.open(arc_25)[1].data
arc_3 = fits.open(arc_3)[1].data
arc_4 = fits.open(arc_4)[1].data
arc_5 = fits.open(arc_5)[1].data
arc_6 = fits.open(arc_6)[1].data

# Create array of number of matches

matches = []
for i in [arc_1, arc_15, arc_2, arc_25, arc_3, arc_4, arc_5, arc_6]:
    matches.append(len(i))

# Print the number of matches
print(matches)

# Plot the number of matches



plt.plot([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0], matches)
plt.xlabel('Tolerance (arcsec)')
plt.ylabel('Number of matches')
plt.title(f'Number of matches vs. tolerance for {cube}_{HST}')
plt.savefig(f'outputs/number_of_matches_vs_tolerance_{cube}_{HST}.pdf')
