
from numpy import loadtxt
import os
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import sys
import numpy as np
import pandas as pd
from astropy.table import Table, Column, MaskedColumn, pprint

"""
Input the matched catalogs are:
muse: outputs/source_catalog_MUSE_{muse_no}_{band}.fits

HST: outputs/source_catalog_HST_{muse_no}_{band}.fits


Changeable variables at the top 

outputs the matched catalog to a new FITS file: outputs/matched_catalog_{muse_no}_{band}_{tolerance_arcsec}.fits
outputs the offsets to a text file: outputs/offsets_{muse_no}_{band}_{tolerance_arcsec}.txt
"""


# Change directory to the location of the FITS file
os.chdir('/home/apatrick/Code')  # Adjust this path if necessary

cube = 28
HST = 814
tolerance_arcsec = 2.0  # tolerance for crossmatching in arcseconds

matched_catalog = fits.open(f'outputs/matched_catalog_{cube}_{HST}_{tolerance_arcsec}.fits')[1].data

# Print the length of the matched catalog
print (len(matched_catalog))

# List of labels to remove
labels_to_remove = [11,13,19,24,34,50,28,51]


# Filter out the rows where cat1_label is in labels_to_remove
vis_filtered_catalog = matched_catalog[~np.isin(matched_catalog['cat1_label'], labels_to_remove)]

# Print the length of the filtered catalog
print(len(vis_filtered_catalog))

# Save the matched catalog to a new FITS file
hdu = fits.BinTableHDU(vis_filtered_catalog)
hdu.writeto(f'outputs/vis_filtered_catalog_{cube}_{HST}_{tolerance_arcsec}.fits', overwrite=True)

