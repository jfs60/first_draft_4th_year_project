from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt
from create_cells import plot_spherocylinder, generate_spherocylinder_mask, poisson_noise_inside_outline
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def spatial_organisation_line(mask, image, horizontal_line, vertical_line): 
    shape = (6,100000)
    new_coordinate_system_image  = np.zeros(shape)

    pruned_skeleton = skeletonize(mask, method = "lee")
    y_coords, x_coords = np.where(mask == 1)

    max_x = np.max(x_coords)
    min_x = np.min(x_coords)
    pruned_skeleton[:, :min_x+5] = 0  # Zero out columns to the left of min_x
    pruned_skeleton[:, max_x-5:] = 0
    y_coords_pruned, x_coords_pruned = np.where(pruned_skeleton ==1 )

    # Perform a quadratic fit (polynomial of order 2)
    coefficients = np.polyfit(x_coords_pruned, y_coords_pruned, 2)

    # Generate the quadratic function
    quadratic_fit = np.poly1d(coefficients)

    # Generate values for the fit line
    x_fit = np.linspace(min_x, max_x, 50)  # Generate 100 points between min and max of x
    y_fit = quadratic_fit(x_fit)
    stick_counter = 0
    for i in range (len(x_fit)-1): 
        count = 1
        temp_x = x_fit[i]
        px = int(temp_x)
        temp_y = y_fit[i]
        py = int(temp_y)
        dy = (y_fit[i]-y_fit[i+1])
        dx = (x_fit[i]-x_fit[i+1])
        centerline_angle = np.arctan2(dy, dx)
        if dx == 0:  # Vertical line
            perp_angle = np.pi / 2
        else:
            centerline_angle = np.arctan2(dy, dx)
            perp_angle = centerline_angle + np.pi / 2
        while mask[py, px] == 1:
            new_coordinate_system_image[3, stick_counter] = image[py, px]
            new_coordinate_system_image[0, stick_counter] = count 
            new_coordinate_system_image[1, stick_counter] = i+1
            new_coordinate_system_image[2, stick_counter] = -1
            new_coordinate_system_image[4, stick_counter] = px 
            new_coordinate_system_image[5, stick_counter] = py 
            temp_x = temp_x + 1* np.cos(perp_angle)
            temp_y = temp_y + 1* np.sin(perp_angle)
            px = int(temp_x)
            py = int(temp_y)
            count += 1
            stick_counter += 1 
        count = 1
        temp_x = x_fit[i]
        px = int(temp_x)
        temp_y = y_fit[i]
        py = int(temp_y)
        while mask[py, px] == 1:
    
            new_coordinate_system_image[3, stick_counter] = image[py, px]
            new_coordinate_system_image[0, stick_counter] = count 
            new_coordinate_system_image[1, stick_counter] = i+1
            new_coordinate_system_image[2, stick_counter] = 1
            new_coordinate_system_image[4, stick_counter] = px 
            new_coordinate_system_image[5, stick_counter] = py 
            temp_x = temp_x - 1* np.cos(perp_angle)
            temp_y = temp_y - 1* np.sin(perp_angle)
            px = int(temp_x)
            py = int(temp_y)
            count += 1 
            stick_counter +=1
    analyse_this = new_coordinate_system_image[:, :(np.count_nonzero(new_coordinate_system_image[3, :]))]

    r_values = analyse_this[0]
    l_values = analyse_this[1]
    phi_values = analyse_this[2]
    pixel_values = analyse_this[3]
    x_values = analyse_this[4].astype(int)
    y_values = analyse_this[5].astype(int)
    valid_indices = mask[y_values, x_values] != 0 
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]
    r_values = r_values[valid_indices]
    l_values = l_values[valid_indices]
    phi_values = phi_values[valid_indices]
    pixel_values = pixel_values[valid_indices]

    l_values_include_phi = l_values*phi_values
    r_values_include_phi = r_values*phi_values

    label = horizontal_line
    label_1 = vertical_line
    label_2 = -vertical_line
    label_values_r = pixel_values[r_values_include_phi == label]
    label_values_l_positive = pixel_values[(l_values_include_phi == label_1)]
    label_values_l_negative = pixel_values[(l_values_include_phi == label_2)]
    x_values_r = x_values[r_values_include_phi == label]
    x_values_l = x_values[(l_values_include_phi == label_1) | (l_values_include_phi == label_2)]

    y_values_r = y_values[r_values_include_phi == label]
    y_values_l = y_values[(l_values_include_phi == label_1) | (l_values_include_phi == label_2)]


    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    axes[0,0].imshow(mask, cmap='gray')
    axes[0,0].imshow(image, cmap='gray',alpha = 0.5)
    axes[0,0].set_title('Spatial Organisation Analysed Across l')
    axes[0,0].axis('off')
    axes[0,0].plot(x_fit, y_fit, label='Centerline', color='blue')
    axes[0,0].plot(x_values_r, y_values_r,label='Line Analysed', color='red' )
    axes[1, 0].set_xlabel('l value')
    axes[1, 0].set_ylabel('pixel value')
    axes[0,0].legend(loc='upper left')

    axes[1,0].plot(label_values_r,label=f"Label {label}", color='b')
    axes[1, 0].grid(True)

    x_positive = np.arange(len(label_values_l_positive))  
    x_negative = -np.arange(len(label_values_l_negative)) 

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0,1].imshow(image, cmap='gray',alpha = 0.5)
    axes[0, 1].plot()
    axes[0, 1].set_title('Spatial Organisation Analysed Across r')
    axes[0, 1].axis('off')
    axes[0, 1].plot(x_fit, y_fit, label='Centerline', color='blue')
    axes[0,1].plot(x_values_l, y_values_l, label='Line Analysed', color='red' )
    axes[0,1].legend(loc='upper left')

    axes[1,1].plot(x_positive,label_values_l_positive, color='b')   
    axes[1,1].plot(x_negative,label_values_l_negative, color='b')   
    axes[1, 1].set_xlabel('r value')
    axes[1, 1].set_ylabel('pixel value')
    axes[1, 1].grid(True)


    plt.tight_layout()
    plt.show()

def spatial_organisation_along_length (mask, image): 
    shape = (6,100000)
    new_coordinate_system_image  = np.zeros(shape)

    pruned_skeleton = skeletonize(mask, method = "lee")
    y_coords, x_coords = np.where(mask == 1)

    max_x = np.max(x_coords)
    min_x = np.min(x_coords)
    pruned_skeleton[:, :min_x+5] = 0  # Zero out columns to the left of min_x
    pruned_skeleton[:, max_x-5:] = 0
    y_coords_pruned, x_coords_pruned = np.where(pruned_skeleton ==1 )

    # Perform a quadratic fit (polynomial of order 2)
    coefficients = np.polyfit(x_coords_pruned, y_coords_pruned, 2)

    # Generate the quadratic function
    quadratic_fit = np.poly1d(coefficients)

    # Generate values for the fit line
    x_fit = np.linspace(min_x, max_x, 50)  # Generate 100 points between min and max of x
    y_fit = quadratic_fit(x_fit)
    stick_counter = 0
    for i in range (len(x_fit)-1): 
        count = 1
        temp_x = x_fit[i]
        px = int(temp_x)
        temp_y = y_fit[i]
        py = int(temp_y)
        dy = (y_fit[i]-y_fit[i+1])
        dx = (x_fit[i]-x_fit[i+1])
        centerline_angle = np.arctan2(dy, dx)
        if dx == 0:  # Vertical line
            perp_angle = np.pi / 2
        else:
            centerline_angle = np.arctan2(dy, dx)
            perp_angle = centerline_angle + np.pi / 2
        while mask[py, px] == 1:
            new_coordinate_system_image[3, stick_counter] = image[py, px]
            new_coordinate_system_image[0, stick_counter] = count 
            new_coordinate_system_image[1, stick_counter] = i+1
            new_coordinate_system_image[2, stick_counter] = -1
            new_coordinate_system_image[4, stick_counter] = px 
            new_coordinate_system_image[5, stick_counter] = py 
            temp_x = temp_x + 1* np.cos(perp_angle)
            temp_y = temp_y + 1* np.sin(perp_angle)
            px = int(temp_x)
            py = int(temp_y)
            count += 1
            stick_counter += 1 
        count = 1
        temp_x = x_fit[i]
        px = int(temp_x)
        temp_y = y_fit[i]
        py = int(temp_y)
        while mask[py, px] == 1:
    
            new_coordinate_system_image[3, stick_counter] = image[py, px]
            new_coordinate_system_image[0, stick_counter] = count 
            new_coordinate_system_image[1, stick_counter] = i+1
            new_coordinate_system_image[2, stick_counter] = 1
            new_coordinate_system_image[4, stick_counter] = px 
            new_coordinate_system_image[5, stick_counter] = py 
            temp_x = temp_x - 1* np.cos(perp_angle)
            temp_y = temp_y - 1* np.sin(perp_angle)
            px = int(temp_x)
            py = int(temp_y)
            count += 1 
            stick_counter +=1
    analyse_this = new_coordinate_system_image[:, :(np.count_nonzero(new_coordinate_system_image[3, :]))]

    r_values = analyse_this[0]
    l_values = analyse_this[1]
    phi_values = analyse_this[2]
    pixel_values = analyse_this[3]
    x_values = analyse_this[4].astype(int)
    y_values = analyse_this[5].astype(int)
    valid_indices = mask[y_values, x_values] != 0 
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]
    r_values = r_values[valid_indices]
    l_values = l_values[valid_indices]
    phi_values = phi_values[valid_indices]
    pixel_values = pixel_values[valid_indices]

    l_values_include_phi = l_values*phi_values
    r_values_include_phi = r_values*phi_values

    max_l = np.max(l_values_include_phi)
    min_l = np.min(l_values_include_phi)
    label_1 = np.arange(0, max_l, 4) 
    label_2 = np.arange(0, min_l, -4)
    x_values_l = []
    y_values_l = []

    x_positive = []
    x_negative = []

    label_values_l_negative =[]
    label_values_l_positive =[]

    fig, axes = plt.subplots(2, 1 , figsize=(10, 5))
    for i in range (len(label_1)):
        temp_l_positive = (pixel_values[(l_values_include_phi == label_1[i])])
        temp_l_negative = (pixel_values[(l_values_include_phi == label_2[i])])
        label_values_l_positive.append(temp_l_positive)
        label_values_l_negative.append(temp_l_negative)
        x_values_l.append(x_values[(l_values_include_phi == label_1[i]) | (l_values_include_phi == label_2[i])])

        y_values_l.append(y_values[(l_values_include_phi == label_1[i]) | (l_values_include_phi == label_2[i])])

        x_positive.append(np.arange(len(temp_l_positive))) 
        x_negative.append(-np.arange(len(temp_l_negative)))
        
    axes[0].imshow(mask, cmap='gray')
    axes[0].plot()
    axes[0].set_title('Spatial Organisation Analysed Across r')
    axes[0].axis('off')
    axes[0].plot(x_fit, y_fit, label='Centerline', color='blue')
    axes[0].imshow(image, alpha = 0.5)
    for i in range(len(x_values_l)):

        axes[0].plot(x_values_l[i], y_values_l[i], label='Line Analysed', color='red' )

    average_combined = [
        np.nanmean([np.mean(pos), np.mean(neg)]) if len(pos) > 0 and len(neg) > 0 else np.nan
        for pos, neg in zip(label_values_l_positive, label_values_l_negative)
    ]

    # X-axis is the index i
    i_values = np.arange(len(average_combined))
    # Plot the average pixel values against i
    axes[1].plot(i_values, average_combined, label="Pixel Value Average for mesh", color='b', marker='o')

    # Formatting
    axes[1].set_xlabel('mesh value')
    axes[1].set_ylabel('Average Pixel Value')
    axes[1].set_title('Average Pixel Intensity across i')
    axes[1].grid(True)
    axes[1].legend()


    
    axes[1].set_xlabel('r value')
    axes[1].set_ylabel('pixel value')
    axes[1].grid(True)



    plt.tight_layout()
    plt.show()








