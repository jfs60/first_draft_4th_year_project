import numpy as np
from skimage import measure
from pyefd import elliptic_fourier_descriptors
import mahotas
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import imageio.v3 as iio



try:
    _range = xrange
except NameError:
    _range = range


def efd_cost_function_plot(image, error_term_used = "AIC", no_of_point = 300, no_of_orders_evaluated = 20): 
    AIC_list = []
    BIC_list = []
    y_coords, x_coords = np.nonzero(image)
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    locus = (mean_x, mean_y)
    coeffs_sum_squared = 0
    contours = measure.find_contours(image, 0.8)
    coeffs = elliptic_fourier_descriptors(contours[0], order=no_of_orders_evaluated, normalize=False)
    t = np.linspace(0, 1.0, no_of_point)
    xt = np.ones((no_of_point,)) * locus[1] #to make it start from the centre point of the image
    yt = np.ones((no_of_point,)) * locus[0]
    image = (image > image.mean())

    # Label connected components
    labelled, n_nucleus = mahotas.label(image)
    relabelled = mahotas.bwperim(labelled)

    x_coords, y_coords = np.where(relabelled)
    points_bw_perim = np.array(list(zip(y_coords,x_coords)))

    for n in _range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )
        coeffs_sum_squared += (coeffs[n, 0])**2 + coeffs[n, 1]**2 + coeffs[n, 2]**2 + coeffs[n, 3]**2

        points = np.array(list(zip(yt,xt)))
        # Build a KDTree for efficient nearest-neighbor search
        tree = cKDTree(points)

        # Find the nearest neighbor for each point in the first dataset
        distances, indices = tree.query(points_bw_perim)

        # Calculate the total squared error
        total_squared_error = np.sum(distances ** 2)
        k = 4 * (n + 1)  # Degrees of freedom
        mse = total_squared_error / no_of_point

        AIC_penalty_term = 2*k
        BIC_penalty_term = k*np.log(no_of_point)
        AIC_error_term =no_of_point * np.log(mse)
        BIC_error_term = no_of_point * np.log(mse)
        
        AIC = 2 * k + no_of_point * np.log(mse)
        BIC = k * np.log(no_of_point) + no_of_point * np.log(mse)
        
        AIC_list.append(AIC)
        BIC_list.append(BIC)

    
    x_values = np.linspace(1, 20, 20)
    if error_term_used == "AIC": 
        plt.plot(x_values, AIC_list)
        plt.xticks(np.arange(1, 21, 1))
        plt.title("AIC Plot")
        plt.show()
    elif error_term_used == "BIC": 
        plt.plot(x_values, BIC_list)
        plt.xticks(np.arange(1, 21, 1))
        plt.title("BIC Plot")
        plt.show()

def find_optimal_efd(image, error_term_used = "AIC", no_of_point = 300, no_of_orders_evaluated = 20, show_plot = False):
    
    y_coords, x_coords = np.nonzero(image)
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    locus = (mean_x, mean_y)
    coeffs_sum_squared = 0
    contours = measure.find_contours(image, 0.8)
    coeffs = elliptic_fourier_descriptors(contours[0], order=no_of_orders_evaluated, normalize=False)
    t = np.linspace(0, 1.0, no_of_point)
    xt = np.ones((no_of_point,)) * locus[1] #to make it start from the centre point of the image
    yt = np.ones((no_of_point,)) * locus[0]
    image = (image > image.mean())
    optimal_cost_value = 9999999
    optimal_cost_order = -1
    optimal_xt = np.ones((no_of_point,)) * locus[1]
    optimal_yt = np.ones((no_of_point,)) * locus[0]

    # Label connected components
    labelled, n_nucleus = mahotas.label(image)
    relabelled = mahotas.bwperim(labelled)

    x_coords, y_coords = np.where(relabelled)
    points_bw_perim = np.array(list(zip(y_coords,x_coords)))

    for n in _range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )
        coeffs_sum_squared += (coeffs[n, 0])**2 + coeffs[n, 1]**2 + coeffs[n, 2]**2 + coeffs[n, 3]**2

        points = np.array(list(zip(yt,xt)))
        # Build a KDTree for efficient nearest-neighbor search
        tree = cKDTree(points)

        # Find the nearest neighbor for each point in the first dataset
        distances, indices = tree.query(points_bw_perim)

        # Calculate the total squared error
        total_squared_error = np.sum(distances ** 2)
        k = 4 * (n + 1)  # Degrees of freedom
        mse = total_squared_error / no_of_point

        AIC_penalty_term = 2*k
        BIC_penalty_term = k*np.log(no_of_point)
        AIC_error_term =no_of_point * np.log(mse)
        BIC_error_term = no_of_point * np.log(mse)
        AIC = 2 * k + no_of_point * np.log(mse)
        BIC = k * np.log(no_of_point) + no_of_point * np.log(mse)
        if error_term_used == "AIC":
            if AIC < optimal_cost_value: 
                optimal_cost_value = AIC 
                optimal_cost_order = n+1
                optimal_xt = xt 
                optimal_yt = yt
        elif error_term_used =="BIC": 
            if BIC < optimal_cost_value: 
                optimal_cost_value = BIC
                optimal_cost_order = n+1
                optimal_xt = xt 
                optimal_yt = yt

        
    if show_plot == True: 
        plt.imshow(image)
        plt.plot(optimal_yt,optimal_xt)
    return optimal_cost_order, optimal_xt, optimal_yt



# masks =iio.imread("synthetic_data/masks/Nonesynth_00000.png")
# test_mask = (np.rot90(masks, 1) ==(1)).astype(np.uint8)
 

# efd_cost_function_plot(test_mask)
# optimal_cost_order, optimal_xt, optimal_yt = find_optimal_efd(test_mask )
# print(optimal_cost_order)