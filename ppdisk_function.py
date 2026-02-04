import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

# Clip the center region of the image
def clip_center(data, region_size):
    ny, nx = data.shape
    cy, cx = ny // 2, nx // 2
    half_size = region_size // 2
    return data[cy - half_size:cy + half_size, cx - half_size:cx + half_size]


#Calculate azimuthal profile

def calculate_azimuthal_profile(data, center_x, center_y, min_radius, max_radius, num_bins=72):
    """
    Calculate the azimuthal profile of the data.
    
    Parameters:
    - data: 2D numpy array of the disk data.
    - center_x: x-coordinate of the center of the disk.
    - center_y: y-coordinate of the center of the disk.
    - min_radius: Minimum radius for the annular region.
    - max_radius: Maximum radius for the annular region.
    - num_bins: Number of bins for the azimuthal profile (default is 72 for 5-degree bins).
    
    Returns:
    - bin_centers: The azimuthal bin centers.
    - profile: The averaged profile values.
    - profile_std: The standard deviation of the profile values.
    """
    
    Y, X = np.ogrid[:data.shape[0], :data.shape[1]]
    rel_x = X - center_x
    rel_y = Y - center_y

    r = np.sqrt(rel_x**2 + rel_y**2)
    theta = np.arctan2(rel_y, rel_x)

    # Adjust theta to start from the y-axis by adding π/2 radians
    theta_adjusted = theta - np.pi / 2

    # Ensure theta_adjusted is within the range [0, 2π)
    theta_adjusted = np.mod(theta_adjusted, 2 * np.pi)

    # Convert adjusted theta from radians to degrees
    theta_degrees = np.rad2deg(theta_adjusted)

    annular_mask = (r >= min_radius) & (r <= max_radius)
    values = data[annular_mask]
    theta_values = theta_degrees[annular_mask]

    bin_edges = np.linspace(0, 360, num_bins + 1)
    #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    profile = np.zeros(num_bins)
    profile_std = np.zeros(num_bins)

    for i in range(num_bins):
        min_angle = bin_edges[i] - 2.5
        max_angle = bin_edges[i + 1] - 2.5
        #min_angle = bin_centers[i] - 180/num_bins
        #max_angle = bin_centers[i] + 180/num_bins
        if min_angle < 0:
            bin_mask = (theta_values >= min_angle + 360) | (theta_values < max_angle)
        elif max_angle >= 360:
            bin_mask = (theta_values >= min_angle) | (theta_values < max_angle - 360)
        else:
            bin_mask = (theta_values >= min_angle) & (theta_values < max_angle)
        
        bin_values = values[bin_mask]
        profile[i] = np.nanmean(bin_values) if np.any(bin_mask) else np.nan
        profile_std[i] = np.nanstd(bin_values) if np.any(bin_mask) else np.nan

    return profile, profile_std

# Example usage:
# Assuming 'disk_data' is a 2D numpy array representing the protoplanetary disk
# center_x, center_y are the coordinates of the center of the disk
# min_radius, max_radius define the annular region of interest
# bin_centers, profile, profile_std = calculate_azimuthal_profile(disk_data, center_x, center_y, min_radius, max_radius)


# Example usage:
# Assuming 'disk_data' is a 2D numpy array representing the protoplanetary disk
# center_x, center_y are the coordinates of the center of the disk
# min_radius, max_radius define the annular region of interest
# bin_centers, profile, profile_std = calculate_azimuthal_profile(disk_data, center_x, center_y, min_radius, max_radius)



def scale_image_by_r_squared(image, center):
    """
    Scales the intensity values of an image by r^2, where r is the radial distance from a specified center.

    Parameters:
    - image: A 2D NumPy array representing the image.
    - center: A tuple (x, y) specifying the center of the disk in image coordinates.

    Returns:
    - A 2D NumPy array with the intensity values scaled by r^2.
    """
    scaled_image = np.zeros_like(image)
    rows, cols = image.shape

    for x in range(rows):
        for y in range(cols):
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            scaled_image[x, y] = image[x, y] * r**2

    return scaled_image

def deproject_disk(data, inclination_deg, position_angle_deg):
    # Load the FITS file
    #with fits.open(fits_file) as hdul:
    #    data = hdul[0].data
    position_angle_deg=position_angle_deg-90 #correct it from the definition of astronomy
    # Step 1: Rotate the image to align the major axis
    # The position angle is measured from north (up in the image) through east (left in the image)
    # So, we need to rotate by 90 - position_angle to align the major axis with the horizontal axis
    rotated_data = rotate(data, position_angle_deg, reshape=True)


    # Step 2: Correct for inclination
    # Stretch the image along the vertical axis
    scale_factor = 1 / np.cos(np.radians(inclination_deg))
    stretched_data = np.zeros((int(rotated_data.shape[0] * scale_factor), rotated_data.shape[1]))
    for i in range(rotated_data.shape[1]):
        stretched_data[:, i] = np.interp(np.linspace(0, rotated_data.shape[0], stretched_data.shape[0]),
                                         np.arange(rotated_data.shape[0]),
                                         rotated_data[:, i])
    
     # Make the image square
    max_dim = max(stretched_data.shape)
    square_image = np.zeros((max_dim, max_dim))
    start_x = (max_dim - stretched_data.shape[1]) // 2
    start_y = (max_dim - stretched_data.shape[0]) // 2
    square_image[start_y:start_y + stretched_data.shape[0], start_x:start_x + stretched_data.shape[1]] = stretched_data

    # Step 3: Optionally rotate back to the original orientation
    # Uncomment the next line if you want to rotate back
    square_image = rotate(square_image, -position_angle_deg, reshape=True)

    return square_image

def calculate_radial_profile(data, r_min, r_max, pa_min, pa_max, center=None, num_bins=100):
    """
    Draw the radial profile of a protoplanetary disk and return the standard deviations.
    
    Parameters:
    - data: 2D numpy array of the disk data.
    - r_min: Minimum radius for the profile.
    - r_max: Maximum radius for the profile.
    - pa_min: Minimum position angle (in degrees) for the profile.
    - pa_max: Maximum position angle (in degrees) for the profile.
    - center: Tuple (x, y) for the center of the disk. If None, the center of the array is used.
    - num_bins: Number of bins for the radial profile.
    
    Returns:
    - r_bin_centers: The radial bin centers.
    - profile: The averaged profile values.
    - profile_std: The standard deviation of the profile values.
    """
    
    # Define the center of the disk
    if center is None:
        center = (data.shape[1] // 2, data.shape[0] // 2)
    
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    pa = np.degrees(np.arctan2(y - center[1], x - center[0]))
    
    # Normalize position angle to be between 0 and 360
    pa = (pa + 360) % 360
    
    # Create a mask for the selected radius and position angle range
    mask = (r >= r_min) & (r <= r_max) & (pa >= pa_min) & (pa <= pa_max)
    
    r_values = r[mask]
    data_values = data[mask]
    
    # Bin the radial values
    r_bins = np.linspace(r_min, r_max, num_bins+1)
    r_bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
    profile = np.zeros(num_bins)
    profile_std = np.zeros(num_bins)
    
    for i in range(num_bins):
        bin_mask = (r_values >= r_bins[i]) & (r_values < r_bins[i+1])
        if np.any(bin_mask):
            profile[i] = np.mean(data_values[bin_mask])
            profile_std[i] = np.std(data_values[bin_mask])
        else:
            profile[i] = np.nan
            profile_std[i] = np.nan
    
    # Plot the radial profile
    #plt.figure(figsize=(8, 6))
    #plt.errorbar(r_bin_centers, profile, yerr=profile_std, label='Radial Profile', fmt='-o')
    #plt.xlabel('Radius')
    #plt.ylabel('Average Intensity')
    #plt.title('Radial Profile of Protoplanetary Disk')
    #plt.legend()
    #plt.grid()
    #plt.show()
    
    return r_bin_centers, profile, profile_std

# Example usage:
# Assuming 'disk_data' is a 2D numpy array representing the protoplanetary disk
# r_bin_centers, profile, profile_std = draw_radial_profile(disk_data, 10, 100, 0, 360)



