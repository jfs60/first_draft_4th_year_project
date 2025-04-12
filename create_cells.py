import matplotlib.pyplot as plt 
import numpy as np 
import cv2
from matplotlib.path import Path


def plot_spherocylinder(L, R, phi):
    """
    Generates a filled 2D projection of a spherocylindrical cell centered at (0,0).
    Generates a binary mask from the outline of the spherocylinder
    
    Parameters:
    L   : Length of the cylindrical (rod) portion.
    R   : Radius of the cell (and of the hemispherical endcaps).
    phi : Bending angle (in radians) for the cylindrical part. Use phi = 0 for a straight cell.

    Returns:
    - outline_x and outline_y
    """
    if abs(phi) < 1e-6:
        # --- Straight Spherocylinder ---
        A = np.array([-L/2, 0])  # left center of endcap
        B = np.array([L/2, 0])   # right center of endcap
        
        # Define edges
        A_top, A_bottom = A + [0, R], A - [0, R]
        B_top, B_bottom = B + [0, R], B - [0, R]
        
        # Top and bottom edges
        top_x = np.linspace(A_top[0], B_top[0], 50)
        top_y = np.linspace(A_top[1], B_top[1], 50)
        bottom_x = np.linspace(B_bottom[0], A_bottom[0], 50)
        bottom_y = np.linspace(B_bottom[1], A_bottom[1], 50)
        
        # Left hemisphere
        theta_left = np.linspace(np.arctan2(A_bottom[1] - A[1], A_bottom[0] - A[0]),
                                 np.arctan2(A_top[1] - A[1], A_top[0] - A[0]), 50)
        left_x = A[0] - R * np.cos(theta_left)
        left_y = A[1] + R * np.sin(theta_left)
        
        
        # Right hemisphere
        theta_right = np.linspace(np.arctan2(B_top[1] - B[1], B_top[0] - B[0]),
                                  np.arctan2(B_bottom[1] - B[1], B_bottom[0] - B[0]), 50)
        right_x = B[0] + R * np.cos(theta_right)
        right_y = B[1] + R * np.sin(theta_right)
        
        # Assemble shape outline
        outline_x = np.concatenate([left_x, top_x, right_x, bottom_x])
        outline_y = np.concatenate([left_y, top_y, right_y, bottom_y])
    
    else:
        # --- Bent Spherocylinder ---
        R_arc = L / phi
        t = np.linspace(-phi/2, phi/2, 100)  # Arc centered at (0,0)
        mid_x = R_arc * np.sin(t)
        mid_y = R_arc - R_arc * np.cos(t)
        
        A, B = np.array([mid_x[0], mid_y[0]]), np.array([mid_x[-1], mid_y[-1]])
        
        T0, T1 = np.array([np.cos(t[0]), np.sin(t[0])]), np.array([np.cos(t[-1]), np.sin(t[-1])])
        N0, N1 = np.array([-np.sin(t[0]), np.cos(t[0])]), np.array([-np.sin(t[-1]), np.cos(t[-1])])
        
        A_top, A_bottom = A + R * N0, A - R * N0
        B_top, B_bottom = B + R * N1, B - R * N1
        
        Nx, Ny = -np.sin(t), np.cos(t)
        offset_top_x = mid_x + R * Nx
        offset_top_y = mid_y + R * Ny
        offset_bottom_x = mid_x - R * Nx
        offset_bottom_y = mid_y - R * Ny
        
        offset_top_x[0], offset_top_y[0] = A_top
        offset_top_x[-1], offset_top_y[-1] = B_top
        offset_bottom_x[0], offset_bottom_y[0] = A_bottom
        offset_bottom_x[-1], offset_bottom_y[-1] = B_bottom
        
        # Left hemisphere
        theta_left = np.linspace(-np.pi / 2, np.pi / 2, 50)
        left_x = A[0] - R * np.cos(theta_left)
        left_y = A[1] + R * np.sin(theta_left)
        
        # Right hemisphere
        theta_right = np.linspace(np.pi / 2, -np.pi / 2, 50)
        right_x = B[0] + R * np.cos(theta_right)
        right_y = B[1] + R * np.sin(theta_right)
        
        outline_x = np.concatenate([left_x, offset_top_x[1:-1], right_x, offset_bottom_x[::-1][1:-1]])
        outline_y = np.concatenate([left_y, offset_top_y[1:-1], right_y, offset_bottom_y[::-1][1:-1]])
    return outline_x, outline_y

def generate_spherocylinder_mask (outline_x, outline_y, pixel_size, margin = 100):
    """    
    Generates a binary mask from the outline of the spherocylinder

    """
    # Step 1: Determine the bounding box of the shape
    min_x, max_x = min(outline_x), max(outline_x)
    min_y, max_y = min(outline_y), max(outline_y)

    # Step 2: Compute the required pixel dimensions
    width_pixels = int(np.ceil((max_x - min_x) / pixel_size)) + 2 * margin
    height_pixels = int(np.ceil((max_y - min_y) / pixel_size)) + 2 * margin

    # Step 3: Scale and shift coordinates into pixel space
    scaled_x = np.round((outline_x - min_x) / pixel_size).astype(int) + margin
    scaled_y = np.round((outline_y - min_y) / pixel_size).astype(int) + margin

    # Step 4: Create an empty mask
    binary_mask = np.zeros((height_pixels, width_pixels), dtype=np.uint8)


    # Step 5: Fill the mask using OpenCV
    pts = np.array([np.column_stack((scaled_x, scaled_y))], dtype=np.int32)
    cv2.fillPoly(binary_mask, pts, 255)
    height, width = binary_mask.shape

    extent = [min_x - pixel_size * margin,
            min_x - pixel_size * margin + width * pixel_size,
            min_y - pixel_size * margin,
            min_y - pixel_size * margin + height * pixel_size]


    return binary_mask, extent

def poisson_noise_inside_outline(outline_x, outline_y, lam=5, num_points=1000):
    """
    Generates Poisson-distributed noise points inside a given 2D shape defined by (outline_x, outline_y).
    
    Parameters:
    outline_x, outline_y : Boundary coordinates of the shape.
    lam : Poisson distribution mean (intensity).
    num_points : Number of candidate points to generate.

    Returns:
    x_points, y_points, intensities : Coordinates and Poisson-distributed intensities of points inside the shape.
    """
    # Create a bounding box around the outline
    xmin, xmax = np.min(outline_x), np.max(outline_x)
    ymin, ymax = np.min(outline_y), np.max(outline_y)
    
    # Define the shape as a closed path
    outline_path = Path(np.column_stack([outline_x, outline_y]))
    
    # Generate candidate points inside the bounding box
    x_candidates = np.random.uniform(xmin, xmax, num_points)
    y_candidates = np.random.uniform(ymin, ymax, num_points)
    
    # Filter points that are inside the outline
    inside_mask = outline_path.contains_points(np.column_stack([x_candidates, y_candidates]))
    x_points, y_points = x_candidates[inside_mask], y_candidates[inside_mask]
    
    # Assign Poisson-distributed intensities
    intensities = np.random.poisson(lam, len(x_points))
    
    return x_points, y_points, intensities

# L, R, phi, pixel_size = 100, 25, np.pi/4, 0.2 
# outline_x, outline_y = plot_spherocylinder(L, R, phi)
# generated_mask, extent = generate_spherocylinder_mask(outline_x, outline_y, 0.2)


# L_inner, R_inner, phi_inner = 50, 13, np.pi/4
# inner_x, inner_y = plot_spherocylinder(L_inner, R_inner, phi_inner)
# x_points, y_points, intensities = poisson_noise_inside_outline(inner_x, inner_y, lam=5, num_points=1000)
# fig, ax = plt.subplots(figsize=(6, 6))
# #checking the data is right 
# # Compute extent so that imshow aligns with scatter
# height, width = generated_mask.shape


# ax.imshow(generated_mask, cmap='gray', extent=extent, origin='lower')
# ax.axis('off')
# ax.scatter(x_points, y_points, c=intensities, cmap='gray', alpha=0.7)
# ax.set_aspect('equal')

# ax.set_xlim(-L, L)
# ax.set_ylim(-L, L)
# plt.show()







