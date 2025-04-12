from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
from efd import find_optimal_efd
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.measure import perimeter
from create_cells import plot_spherocylinder, generate_spherocylinder_mask
from IPython.display import display

def spherical_or_not (major_axis, minor_axis): 
    aspect_ratio = major_axis/minor_axis 
    if aspect_ratio <= 1.5: 
        spherical_image = True 
    else: 
        spherical_image = False 
    
    return spherical_image, aspect_ratio
def find_centreline(image, points): 
    pruned_skeleton = skeletonize(image, method = "lee")
    y_coords, x_coords = np.where(image == 1)

    max_x = np.max(x_coords)
    min_x = np.min(x_coords)
    
    pruned_skeleton[:, :min_x+25] = 0  # Zero out columns to the left of min_x
    pruned_skeleton[:, max_x-25:] = 0

    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1


    y_fit, x_fit = np.where(pruned_skeleton == 1)
    
    coefficients = np.polyfit(x_fit, y_fit, 2)

    poly = np.poly1d(coefficients)

    # Generate x values for plotting the best fit line
    x_fit = np.linspace(x_min, x_max, 100)
    y_fit = poly(x_fit)

    return x_fit, y_fit

def find_surface_area_and_volume (x_fit, y_fit, max_distance, image_analysed): 
    total_volume = 0 
    median_radius = 0 
    total_surface_area = 0 
    radius = []
    for i in range(len(x_fit)-1): 

        dy = (y_fit[i]-y_fit[i+1])
        dx = (x_fit[i]-x_fit[i+1])
        center_x, center_y  = x_fit[i] , y_fit[i]

        
        dy_dx = dy/dx
        if dx == 0:  # Vertical line
            perp_angle = np.pi / 2
        else:
            centerline_angle = np.arctan2(dy, dx)
            perp_angle = centerline_angle + np.pi / 2

        top_distance = bottom_distance = None

    # Check in both directions along the perpendicular line
        for sign in [-1, 1]:  # -1 for one side, 1 for the other
            for dist in range(max_distance):
                check_x = int(center_x + (dist) * np.cos(perp_angle) * sign)
                check_y = int(center_y + (dist) * np.sin(perp_angle) * sign)

                # Ensure coordinates are within image bounds
                if 0 <= check_x < image_analysed.shape[1] and 0 <= check_y < image_analysed.shape[0]:
                    if image_analysed[check_y, check_x] == 0:  # Found the edge
                        if sign == -1:
                            top_distance = dist
                        else:
                            bottom_distance = dist
                        break
            
            move_along_centreline = np.sqrt(dx**2 +dy**2)
            if top_distance is not None and bottom_distance is not None:
                # print("top distance", top_distance)
                # print("bottom distance", bottom_distance)
                radius.append(round(top_distance, 3))
                radius.append(round(bottom_distance, 3))
                width = top_distance + bottom_distance
                total_surface_area += move_along_centreline*width
                total_volume += move_along_centreline*width*width
    median_radius = statistics.median(radius)*0.065/3
    return total_surface_area, total_volume, median_radius

def analyse_cell_morphology (images, multiple_images = False): 
    if multiple_images == True: 
        aspect_ratio_list = []
        optimal_order_efd_list = []
        total_length_centreline_list = []
        total_surface_area_list = []
        median_radius_list = []
        total_volume_list = []
        sa_to_vol_list = []
        perimeter_list = []
        circularity_list = []
        slenderness_list = []
        amount_of_bend_list = []
        label_of_image = []
        for image in images:
            y_coords, x_coords = np.nonzero(image)
            mean_x = np.mean(x_coords)
            mean_y = np.mean(y_coords)
            locus = (mean_x, mean_y)
            # Find horizontal axis length
            left = np.min(x_coords[x_coords < mean_x], initial=mean_x)
            right = np.max(x_coords[x_coords > mean_x], initial=mean_x)
            major_axis_length = right - left
            
            # Find vertical axis length
            top = np.min(y_coords[y_coords < mean_y], initial=mean_y)
            bottom = np.max(y_coords[y_coords > mean_y], initial=mean_y)
            minor_axis_length = bottom - top


            spherical_image, aspect_ratio = spherical_or_not(major_axis_length, minor_axis_length)
            aspect_ratio_list.append(aspect_ratio)
            optimal_cost_order, optimal_xt, optimal_yt = find_optimal_efd(image)
            optimal_order_efd_list.append(optimal_cost_order)
            points = np.array(list(zip(optimal_yt,optimal_xt)))
            if spherical_image == False: 
                degree = 2
                x_fit, y_fit = find_centreline(image, points)
                min_x = np.min(x_fit)
                distances = np.sqrt(np.diff(x_fit)**2 + np.diff(y_fit)**2)
                total_length = np.sum(distances)
                total_length = (total_length*0.065)/3
                total_length_centreline_list.append(total_length)
                best_fit_line = np.array(list(zip(y_fit, x_fit)))
                mid_index = len(x_fit) // 2
                mid_point_x = x_fit[mid_index]
                mid_point_y = y_fit[mid_index]
                a =mid_point_x - min_x
                h = np.sqrt((mid_point_x-min_x)**2 + (mid_point_y-y_fit[0])**2)
                angle_rad = np.arccos(a/h)  # Angle in radians
            else: 
                total_length = math.dist([right, mean_y], [left, mean_y])
                total_length = (total_length*0.065)/3
                total_length_centreline_list.append(total_length)
                x_fit = np.linspace(right, left, 100)
                y_fit = np.linspace(mean_y, mean_y, 100)

            mid_index = len(x_fit) // 2
            mid_point_x = x_fit[mid_index]
            mid_point_y = y_fit[mid_index]
            if min_x == None: 
                break
            
            a =mid_point_x - min_x
            h = np.sqrt((mid_point_x-min_x)**2 + (mid_point_y-y_fit[0])**2)
            angle_rad = np.arccos(a/h)  # Angle in radians
            amount_of_bend_list.append(angle_rad)

            predicted_total_surface_area, predicted_total_volume, median_radius = find_surface_area_and_volume (x_fit, y_fit, 300, image)
            total_surface_area_list.append(predicted_total_surface_area)
            surface_area = np.sum(image == 1)
            median_radius_list.append(median_radius)

            total_volume_list.append(predicted_total_volume)

            sa_to_vol_ratio = predicted_total_surface_area/predicted_total_volume 
            sa_to_vol_list.append(sa_to_vol_ratio)
            image_perimeter = perimeter(image)
            perimeter_list.append(image_perimeter)
            circularity = (4*np.pi*predicted_total_surface_area)/image_perimeter
            circularity_list.append(circularity)
            slenderness = total_length/median_radius
            slenderness_list.append(slenderness)


    data_dict = {
        'Optimal Order EFD': optimal_order_efd_list,
        'Aspect Ratio': aspect_ratio_list,
        'Length of Centreline': total_length_centreline_list,
        'Total Surface Area': total_surface_area_list,
        'Median Radius': median_radius_list,
        'Total Volume': total_volume_list,
        'Surface Area to Vol Ratio': sa_to_vol_list,
        'Perimeter': perimeter_list,
        'Circularity': circularity_list,
        'Slenderness': slenderness_list,
        'amount of bend': amount_of_bend_list
        

    }
    return(data_dict)

# Define parameter lists
L = np.linspace(25, 225, 21)
R = np.linspace(10,25, 6)
print(R)
phi_list = np.pi/4
pixelation_constant = 0.5 

l_list = []
r_list = []
Phi_list = []
counter = 0  # Track index for image storage
binary_mask_list = []

# Loop through predefined values
for l in L:
    for r in R: 
        # Generate spherocylinder outline
        outline_x, outline_y = plot_spherocylinder(l, r, phi_list)
        l_list.append(l)
        r_list.append(r)
        Phi_list.append(phi_list)
        # Create figure and plot
        binary_mask, extent = generate_spherocylinder_mask(outline_x, outline_y, pixelation_constant)
        binary_mask = (binary_mask>128).astype(np.uint8)

        # Store the image
        binary_mask_list.append(binary_mask)
        
        counter += 1  # Increment index

        


data_dict = analyse_cell_morphology(binary_mask_list,True)
for key, value in data_dict.items():
    print(f"{key}: {len(value)}")
# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data_dict)
df = df.map(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)
display(df)