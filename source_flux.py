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
HST = 606
tolerance = 1.5

# Define the paths to the FITS files
catalog_path = f'outputs/matched_catalog_{cube}_{HST}_{tolerance}.fits'


# Load the catalog from the FITS file
with fits.open(catalog_path) as hdul:
    # Access the data and the header of the first extension HDU (index 1)
    catalog_data = hdul[1].data
    catalog_header = hdul[1].header
    
    # Check if the WCS information exists and extract it if present
    if 'WCSAXES' in hdul[0].header:
        wcs = WCS(hdul[0].header)
    else:
        wcs = None  # If WCS is not needed for this particular task
        #print (None)
    
    # Close the FITS file
    hdul.close()

#print(catalog_data.columns)

# plot the kron fluxe in ujy against the hst fluxes in ujy
plt.figure(figsize=(8, 8))
plt.scatter(catalog_data['cat1_kron_flux_uJy'], catalog_data['cat2_kron_flux_uJy'], color='blue', alpha=0.5)
plt.xlabel('MUSE kron flux uJy')
plt.ylabel('HST kron flux uJy')
plt.title(f'Flux Comparison for {cube}_{HST} at {tolerance} arcsec')
plt.savefig(f'outputs/flux_comparison_{cube}_{HST}_{tolerance}.pdf')


# Plot the kron flux for muse and hst as histograms
plt.figure(figsize=(8, 8))
plt.hist(catalog_data['cat1_kron_flux_uJy'], bins=100, color='blue', alpha=0.5, label='MUSE', density=True)
plt.hist(catalog_data['cat2_kron_flux_uJy'], bins=100, color='orange', alpha=0.5, label='HST', density=True)
plt.xlabel('Kron Flux (uJy)')
plt.ylabel('Density')
plt.legend()
plt.title(f'Kron Flux Histogram for {cube}_{HST} at {tolerance} arcsec')
plt.ylim(0, 1)  # Ensure y-axis is limited to 0-1
#plt.xlim(0, 100)
plt.savefig(f'outputs/flux_histogram_{cube}_{HST}_{tolerance}.pdf')

# Print the cube number, HST band and tolerance
print(f'Cube: {cube}, HST Band: {HST}, Tolerance: {tolerance}')

# Print the max and min flux for muse and hst
print ( f'MUSE max flux {max(catalog_data["cat1_kron_flux_uJy"])}')
print ( f'MUSE min flux {min(catalog_data["cat1_kron_flux_uJy"])}')
print ( f'HST max flux {max(catalog_data["cat2_kron_flux_uJy"])}')
print ( f'HST min flux {min(catalog_data["cat2_kron_flux_uJy"])}')