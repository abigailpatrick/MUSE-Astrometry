from functions import crossmatch_catalogs, add_offsets, plot_offset
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

cube = 27
HST = 606

tolerance_arcsec = 2.0  # tolerance for crossmatching in arcseconds
kron_area_exlusion = 30  # exclusion for kron area
HST_faint_exclusion = 0.1  # exclusion for faint HST sources
fwhm_limit = 22 # exclusion for fwhm


# Define the paths to the FITS files
catalog1_path = f'outputs/source_catalog_MUSE_{cube}_{HST}.fits'
catalog2_path = f'outputs/source_catalog_HST_{cube}_{HST}.fits'

 # Load the catalogs from FITS files
catalog1 = fits.open(catalog1_path)[1].data
print (len(catalog1))
catalog2 = fits.open(catalog2_path)[1].data

# Crossmatch the catalogs
matched_catalog = crossmatch_catalogs(catalog1, catalog2, tolerance_arcsec)

# Add offsets to the matched catalog
matched_catalog = add_offsets(matched_catalog)

# Print the matched catalog
#print(matched_catalog)

# Print column names
#print(matched_catalog.colnames)

"""" 

# Print the min and max  and 10th and 90th percentiles of kron_radius_arcsec
print(f'Minimum kron radius: {matched_catalog["cat1_kron_radius_arcsec"].min()}')
print(f'Maximum kron radius: {matched_catalog["cat1_kron_radius_arcsec"].max()}')
print(f'10th percentile of kron radius: {np.percentile(matched_catalog["cat1_kron_radius_arcsec"], 10)}')
print(f'80th percentile of kron radius: {np.percentile(matched_catalog["cat1_kron_radius_arcsec"], 80)}')

# Print the min and max  and 10th and 90th percentiles of kron area
print(f'Minimum kron area: {matched_catalog["cat1_aperture_area"].min()}')
print(f'Maximum kron area: {matched_catalog["cat1_aperture_area"].max()}')
print(f'10th percentile of kron area: {np.percentile(matched_catalog["cat1_aperture_area"], 10)}')
print(f'80th percentile of kron area: {np.percentile(matched_catalog["cat1_aperture_area"], 80)}')
"""
# Print the min and max  and 10th and 90th percentiles of fwhm
print(f'Minimum fwhm: {matched_catalog["cat2_fwhm"].min()}')
print(f'Maximum fwhm: {matched_catalog["cat2_fwhm"].max()}')
print(f'10th percentile of fwhm: {np.percentile(matched_catalog["cat2_fwhm"], 10)}')
print(f'80th percentile of fwhm: {np.percentile(matched_catalog["cat2_fwhm"], 80)}')



# Exclude faint HST sources from catalog 
matched_catalog = matched_catalog[matched_catalog['cat2_kron_flux_uJy'] > HST_faint_exclusion]

# Exclude the largest kron radii from the catalog
#matched_catalog = matched_catalog[matched_catalog['cat1_kron_radius_arcsec'] < 1.0]

# Exclude the largest fwhm from the catalog
matched_catalog = matched_catalog[matched_catalog['cat2_fwhm'] < fwhm_limit]


# Exclude the largest kron area from the catalog
#matched_catalog = matched_catalog[matched_catalog['cat1_aperture_area'] < kron_area_exlusion]



# Plot the offsets
plot_offset(matched_catalog, fout=f'outputs/offsets_{cube}_{HST}_{tolerance_arcsec}.pdf')

# Print the label 
print(f'Offsets for {cube} and {HST} at {tolerance_arcsec} arcsec')

# Print the number of matches
print(f'Number of matches: {len(matched_catalog)}')

# Print mean, median and percentile offsets
#print(f'Mean RA offset: {matched_catalog["RA_offset_arcsec"].mean()}')
#print(f'Mean Dec offset: {matched_catalog["Dec_offset_arcsec"].mean()}')
median_ra_offset = np.median(matched_catalog["RA_offset_arcsec"])
print(f'Median RA offset: {median_ra_offset}')
median_dec_offset = np.median(matched_catalog["Dec_offset_arcsec"])
print(f'Median Dec offset: {median_dec_offset}')
print(f'RA offset 16th percentile: {np.percentile(matched_catalog["RA_offset_arcsec"], 16)}')
print(f'Dec offset 16th percentile: {np.percentile(matched_catalog["Dec_offset_arcsec"], 16)}')
print(f'RA offset 84th percentile: {np.percentile(matched_catalog["RA_offset_arcsec"], 84)}')
print(f'Dec offset 84th percentile: {np.percentile(matched_catalog["Dec_offset_arcsec"], 84)}')

# Save the offsets to a text file
with open(f'outputs/offsets_{cube}_{HST}_{tolerance_arcsec}.txt', 'w') as f:
    f.write(f'Number of matches: {len(matched_catalog)}\n')
    f.write(f'Median RA offset: {median_ra_offset}\n')
    f.write(f'Median Dec offset: {median_dec_offset}\n')
    f.write(f'RA offset 16th percentile: {np.percentile(matched_catalog["RA_offset_arcsec"], 16)}\n')
    f.write(f'Dec offset 16th percentile: {np.percentile(matched_catalog["Dec_offset_arcsec"], 16)}\n')
    f.write(f'RA offset 84th percentile: {np.percentile(matched_catalog["RA_offset_arcsec"], 84)}\n')
    f.write(f'Dec offset 84th percentile: {np.percentile(matched_catalog["Dec_offset_arcsec"], 84)}\n')



#print(matched_catalog.colnames)


# Save the matched catalog to a new FITS file
hdu = fits.BinTableHDU(matched_catalog)
hdu.writeto(f'outputs/matched_catalog_{cube}_{HST}_{tolerance_arcsec}.fits', overwrite=True)


"""   

# Remove sources in the top and bottom 10th percentiles of RA and Dec offsets - decide if i want to add to the saved fits file

# Convert the Astropy Table to a pandas DataFrame
matched_catalog_df = matched_catalog.to_pandas()

# Calculate the percentiles for RA and Dec offsets
ra_offset_10th_percentile = np.percentile(matched_catalog_df["RA_offset_arcsec"], 10)
ra_offset_90th_percentile = np.percentile(matched_catalog_df["RA_offset_arcsec"], 90)
dec_offset_10th_percentile = np.percentile(matched_catalog_df["Dec_offset_arcsec"], 10)
dec_offset_90th_percentile = np.percentile(matched_catalog_df["Dec_offset_arcsec"], 90)

# Identify sources in the top and bottom 10th percentiles for both RA and Dec offsets
ra_bottom_10th = matched_catalog_df[matched_catalog_df["RA_offset_arcsec"] <= ra_offset_10th_percentile]
ra_top_10th = matched_catalog_df[matched_catalog_df["RA_offset_arcsec"] >= ra_offset_90th_percentile]
dec_bottom_10th = matched_catalog_df[matched_catalog_df["Dec_offset_arcsec"] <= dec_offset_10th_percentile]
dec_top_10th = matched_catalog_df[matched_catalog_df["Dec_offset_arcsec"] >= dec_offset_90th_percentile]

# Combine the sources from all percentiles
percentile_sources = pd.concat([ra_bottom_10th, ra_top_10th, dec_bottom_10th, dec_top_10th])

# Count the occurrences of each source
source_counts = percentile_sources["cat1_label"].value_counts()

# Identify sources that appear in more than one percentile range
duplicate_sources = source_counts[source_counts > 1].index

# Exclude these sources from the original catalog
filtered_catalog = matched_catalog_df[~matched_catalog_df["cat1_label"].isin(duplicate_sources)]

# Convert filtered catalog back to an Astropy Table
filtered_catalog_table = Table.from_pandas(filtered_catalog)

print(f'Number of sources in the filtered catalog: {len(filtered_catalog)}')
print(f'Number of sources in the original catalog: {len(matched_catalog)}')

median_ra_offset = np.median(filtered_catalog_table["RA_offset_arcsec"])
#print(f'Median RA offset: {median_ra_offset}')
median_dec_offset = np.median(filtered_catalog_table["Dec_offset_arcsec"])
#print(f'Median Dec offset: {median_dec_offset}')

# Save the matched catalog to a new FITS file
#hdu = fits.BinTableHDU(filtered_catalog_table)
#hdu.writeto(f'outputs/matched_catalog_{cube}_{HST}_{tolerance_arcsec}.fits', overwrite=True)



"""