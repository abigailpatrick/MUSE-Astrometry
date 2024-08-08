import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import re


"""
Takes the offsets from the crossmatch and applies them to the WCS information in the MUSE FITS file.

The offsets are from the text files associted with the matched catalogs.

This is giving a seperate update to wcs for each band but there is a potential to do an aversge using extra commented code. 

It outputs a file the same as the muse input to practicerun but wiht updated wcs for ra and dec 
Plus it has the percentiles in the header.


"""

# Set the MUSE cube number
muse_no = 28
band = 814
tolerance_arcsec = 2.0
print(f"MUSE cube number: {muse_no}")

# Function to display header information
def print_header_info(header):
    print("Header Information:")
    for key, value in header.items():
        print(f"{key}: {value}")

# Function to display WCS information
def print_wcs_info(wcs):
    print("\nWCS Information:")
    print(wcs)

# Function to display data information
def print_data_info(data):
    print("\nData Information:")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    if hasattr(data, 'unit'):
        print(f"Data unit: {data.unit}")
    else:
        print("Data unit: Not specified")
    print(f"Data statistics: min={np.min(data)}, max={np.max(data)}, mean={np.mean(data)}")



# Change directory to the location of the FITS file
os.chdir('/home/apatrick/MUSE')  # Adjust this path if necessary

# Load numpy array from csv file
im_data = np.loadtxt(f'im{muse_no}_data.csv', delimiter=',')

# Load the WCS information from the FITS file
with fits.open(f'im{muse_no}_wcs.fits') as hdul:
    im_wcs = WCS(hdul[0].header)
    header = hdul[0].header

# Print header information
print_header_info(header)

# Print WCS information
print_wcs_info(im_wcs)

# Print data information
print_data_info(im_data)

# Change directory to the location of the FITS file
os.chdir('/home/apatrick/Code')

# Read in the offsets and extract values
with open(f'outputs/offsets_{muse_no}_{band}_{tolerance_arcsec}.txt', 'r') as file:
    for line in file:
        # Use regular expressions to find the relevant lines and extract values
        if "Median RA offset" in line:
            median_ra_offset = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Median Dec offset" in line:
            median_dec_offset = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "RA offset 16th percentile" in line:
            ra_offset_16th_percentile = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Dec offset 16th percentile" in line:
            dec_offset_16th_percentile = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "RA offset 84th percentile" in line:
            ra_offset_84th_percentile = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Dec offset 84th percentile" in line:
            dec_offset_84th_percentile = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())

# Print the extracted values
print("Median RA offset:", median_ra_offset)
print("Median Dec offset:", median_dec_offset)
print("RA offset 16th percentile:", ra_offset_16th_percentile)
print("Dec offset 16th percentile:", dec_offset_16th_percentile)
print("RA offset 84th percentile:", ra_offset_84th_percentile)
print("Dec offset 84th percentile:", dec_offset_84th_percentile)


# Convert arcseconds to degrees
median_ra_offset_deg = median_ra_offset / 3600.0
median_dec_offset_deg = median_dec_offset / 3600.0

# Apply the offsets to the WCS reference values
im_wcs.wcs.crval[0] += median_ra_offset_deg
im_wcs.wcs.crval[1] += median_dec_offset_deg

print("\nUpdated WCS Information:")

# Print updated WCS information
print_wcs_info(im_wcs)

# Print data information
print_data_info(im_data)

# Add the the percentiles to the header
header['RA_OFFSET_16TH_PERCENTILE_ARCSEC'] = ra_offset_16th_percentile
header['DEC_OFFSET_16TH_PERCENTILE_ARCSEC'] = dec_offset_16th_percentile
header['RA_OFFSET_84TH_PERCENTILE_ARCSEC'] = ra_offset_84th_percentile
header['DEC_OFFSET_84TH_PERCENTILE_ARCSEC'] = dec_offset_84th_percentile


# Print updated header information
print_header_info(header)

# Save the updated WCS information to a new FITS file:
new_filename = f'im{muse_no}_wcs_{band}_updated.fits'

# Update the original header with the new WCS information
updated_header = im_wcs.to_header()
for key, value in updated_header.items():
    header[key] = value

# Create a new HDU with the updated header and original data
hdu = fits.PrimaryHDU(data=im_data, header=header)

# Write to a new FITS file
hdu.writeto(new_filename, overwrite=True)
print(f"Updated FITS file saved as {new_filename}")



"""
Do again for an average of the offsets for the two bands 



# Read in the offsets and extract values
with open(f'outputs/offsets_{muse_no}_606_{tolerance_arcsec}.txt', 'r') as file:
    for line in file:
        # Use regular expressions to find the relevant lines and extract values
        if "Median RA offset" in line:
            median_ra_offset_606 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Median Dec offset" in line:
            median_dec_offset_606 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "RA offset 16th percentile" in line:
            ra_offset_16th_percentile_606 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Dec offset 16th percentile" in line:
            dec_offset_16th_percentile_606 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "RA offset 84th percentile" in line:
            ra_offset_84th_percentile_606 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Dec offset 84th percentile" in line:
            dec_offset_84th_percentile_606 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())

# Print the extracted values
print("Median RA offset_606:", median_ra_offset_606)
print("Median Dec offset_606:", median_dec_offset_606)
print("RA offset 16th percentile_606:", ra_offset_16th_percentile_606)
print("Dec offset 16th percentile_606:", dec_offset_16th_percentile_606)
print("RA offset 84th percentile_606:", ra_offset_84th_percentile_606)
print("Dec offset 84th percentile_606:", dec_offset_84th_percentile_606)



# Read in the offsets and extract values
with open(f'outputs/offsets_{muse_no}_814_{tolerance_arcsec}.txt', 'r') as file:
    for line in file:
        # Use regular expressions to find the relevant lines and extract values
        if "Median RA offset" in line:
            median_ra_offset_814 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Median Dec offset" in line:
            median_dec_offset_814 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "RA offset 16th percentile" in line:
            ra_offset_16th_percentile_814 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Dec offset 16th percentile" in line:
            dec_offset_16th_percentile_814 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "RA offset 84th percentile" in line:
            ra_offset_84th_percentile_814 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
        elif "Dec offset 84th percentile" in line:
            dec_offset_84th_percentile_814 = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())

# Print the extracted values
print("Median RA offset_814:", median_ra_offset_814)
print("Median Dec offset_814:", median_dec_offset_814)
print("RA offset 16th percentile_814:", ra_offset_16th_percentile_814)
print("Dec offset 16th percentile_814:", dec_offset_16th_percentile_814)
print("RA offset 84th percentile_814:", ra_offset_84th_percentile_814)
print("Dec offset 84th percentile_814:", dec_offset_84th_percentile_814)

# Calculate the average offsets
median_ra_offset_avg = (median_ra_offset_606 + median_ra_offset_814) / 2
median_dec_offset_avg = (median_dec_offset_606 + median_dec_offset_814) / 2

# Print the average offsets
print("\nAverage RA offset:", median_ra_offset_avg)
print("Average Dec offset:", median_dec_offset_avg)

# Convert arcseconds to degrees

median_ra_offset_deg_avg = median_ra_offset_avg / 3600.0
median_dec_offset_deg_avg = median_dec_offset_avg / 3600.0

# Calculate the average percentiles  - might not be a good method 
ra_offset_16th_percentile_avg = (ra_offset_16th_percentile_606 + ra_offset_16th_percentile_814) / 2
dec_offset_16th_percentile_avg = (dec_offset_16th_percentile_606 + dec_offset_16th_percentile_814) / 2
ra_offset_84th_percentile_avg = (ra_offset_84th_percentile_606 + ra_offset_84th_percentile_814) / 2
dec_offset_84th_percentile_avg = (dec_offset_84th_percentile_606 + dec_offset_84th_percentile_814) / 2


# now apply as above 

"""