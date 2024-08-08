
from functions import source_catalog, visulisation, get_wcs_info, create_cutout,plot_cutout
from functions import convolve_image, plot_cutout_convolved, source_catalog_HST, visulisation_HST
from functions import crossmatch_catalogs, plot_flux_vs_aperture
from numpy import loadtxt
import os
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import sys

"""
Pick the muse catalog and hst image to compare to and create a source catalog for each.
This uses the functions in functions.py to create the source catalogs.

Following this, input the matched catalogs into crossmatch_catalogs to find the sources that are in both catalogs

Most vaiarbles changebale are at the top of the code, others listed below
note in the createion of the source catalog there are variables I have fixed- npixels=10, deblend nlevels=32, 
,deblend contrast=0.001, 2d gaussian kernal,background variables and amount above background is 2 in muse 3 in hst

The muse catalog needs to be in format im28_wcs.fits which is created in create_musefile.ipynb on macbook
The hst image needs to be in format primer_cosmos_acswfc_f814w_30mas_sci.fits.gz


output source catalogs are:
muse: outputs/source_catalog_MUSE_{muse_no}_{band}.fits

HST: outputs/source_catalog_HST_{muse_no}_{band}.fits
"""


# INPUTS
muse_no = 28
#int(sys.argv[1])+27
band = 606

# For HST Convolution
gamma = 2.5 
fwhm_arcs = 0.3   # 0.7 from https://astro.dur.ac.uk/~ams/MUSEcubes/MUSE_OII_short.pdf but ken says 0.3      - seeing fwhm?

radii=[1.0, 2.0, 3.0] # radii for aperture photometry, doesn't matter unless using for light curves

# Change directory to the location of the FITS file
os.chdir('/home/apatrick/MUSE')  # Adjust this path if necessary


MUSE_fits_path = f'/home/apatrick/MUSE/im{muse_no}_wcs.fits' # add 814 for the 814 band 

# Load the updated FITS file
with fits.open(MUSE_fits_path) as hdul:
    # Extract the data and header
    im_data = hdul[0].data
    header = hdul[0].header
    
    # Extract the WCS information
    im_wcs = WCS(header)

# Change directory to the location of the FITS file
os.chdir('/home/apatrick/HST')  # Adjust this path if necessary


# Define the path to the FITS file

fits_file = f'primer_cosmos_acswfc_f{band}w_30mas_sci.fits.gz'

# Open the FITS file
with fits.open(fits_file) as hdul:
    # Print the information about the FITS file
    hdul.info()

    # Get the primary header and data
    primary_header = hdul[0].header
    image_data = hdul[0].data

    # Extract PHOTFLAM and PHOTPLAM from the header
    photflam = primary_header['PHOTFLAM']
    photplam = primary_header['PHOTPLAM']

    # Print the values to confirm
    print(f'PHOTFLAM: {photflam}')
    print(f'PHOTPLAM: {photplam}')

    # Get the WCS information
    wcs_hst = WCS(primary_header)



# Change directory to the location of the FITS file
os.chdir('/home/apatrick/Code')  

# Get the WCS information from the MUSE image
shape_muse, wcs_muse, central_pixel, ra_muse, dec_muse, width_deg, height_deg, pixscale_nyquist = get_wcs_info(im_data, im_wcs) 


# Create a MUSE Soure Catalog
data, tbl, segment_map, segm_deblend, cat, aperture_phot_tbl = source_catalog(im_data,wcs_muse,photplam,pixscale_nyquist,npixels=10, radii=radii,fout=f'outputs/source_catalog_MUSE_{muse_no}_{band}.fits')

visulisation(segment_map, data, segm_deblend, cat, fout=f'outputs/visualisation_{muse_no}_{band}.pdf')


#print(source_catalog(im26_data,10)[1])


# Create a cutout of the HST image
cutout, cutout_wcs = create_cutout(image_data, wcs_hst, width_deg, height_deg, ra_muse, dec_muse, fout=f'outputs/hst_cutout{muse_no}_{band}.fits')


# Plot the cutout of the HST image
plot_cutout(cutout, cutout_wcs,fout=f'outputs/hst_cutout{muse_no}_{band}.pdf') 


# Convolve the HST image with a Moffat kernel
fwhm_pix = fwhm_arcs/pixscale_nyquist.value
convolved_image = convolve_image(cutout, fwhm_pix, gamma)

# Plot the convolved cutout of the HST image
plot_cutout_convolved(convolved_image, fout=f'outputs/hst_convolved_cutout_{muse_no}_{band}.pdf')


#data1, tbl1, segment_map1, segm_deblend1, cat1, aperture_phot_tbl1 = source_catalog_HST(cutout, cutout_wcs, photflam, photplam ,npixels=10, radii=[1.0,2.0,3.0,4.0,5.0,6.0,7.0],fout=f'outputs/source_catalog_HST_{muse_no}_{band}_raw.fits')

#print (f'Number of sources in MUSE image: {len(tbl)}')
#print (f'Number of sources in HST raw image: {len(tbl1)}')

data, tbl, segment_map, segm_deblend, cat, aperture_phot_tbl = source_catalog_HST(convolved_image, cutout_wcs, photflam, photplam ,npixels=10, radii= radii,fout=f'outputs/source_catalog_HST_{muse_no}_{band}.fits')

#print (f'Number of sources in HST convolved image: {len(tbl)}')
print (f'Muse = {muse_no}, HST = {band}')

visulisation_HST(segment_map, data, segm_deblend, cat,fout=f'outputs/visualisation_HST_{muse_no}_{band}.pdf')

#plot_flux_vs_aperture(tbl,tbl1, radii=[1.0,2.0,3.0,4.0,5.0,6.0,7.0],source_index=40, fout=f'outputs/flux_vs_aperture_{muse_no}.pdf')


hdul.close()


