from functions import crossmatch_catalogs, add_offsets, plot_offset_comp
from numpy import loadtxt
import os
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import sys
import numpy as np
import pandas as pd
from astropy.table import Table, Column, MaskedColumn, pprint

cube = 28
HST = 814
tolerance_arcsec = 2.0  # tolerance for crossmatching in arcseconds

mf = 'vis_filtered' # 'matched' or 'vis_filtered'

# Paths 
catalog1_path = f'outputs/{mf}_catalog_{cube}_{HST}_{tolerance_arcsec}.fits'

print (catalog1_path)

# Load the catalogs from FITS files
catalog1 = fits.open(catalog1_path)[1].data

HST = 606

# Paths 
catalog2_path = f'outputs/{mf}_catalog_{cube}_{HST}_{tolerance_arcsec}.fits'

# Load the catalogs from FITS files
catalog2 = fits.open(catalog2_path)[1].data
print (catalog2_path)

# Plot the offset comparison
plot_offset_comp(catalog1, catalog2, fout=f'outputs/{mf}_offset_comparison_{cube}_{tolerance_arcsec}.pdf')

