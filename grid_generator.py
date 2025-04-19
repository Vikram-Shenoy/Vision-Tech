import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance

def divide_isometric_rectangle(image_path, corners=None, grid_size=(4, 6), display=True):
    """
    Divide an isometric rectangle into a grid of equal square cells.
    
    Parameters:
    - image_path: Path to the image containing the isometric rectangle
    - corners: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] defining the rectangle
               If None, the user will need to select the corners manually
    - grid_size: Tuple (rows, cols) specifying the desired grid dimensions
    - display: Whether to display the result
    
    Returns:
    - Original image with grid overlay
    - Grid cell coordinates
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = image.copy()
    
    # If corners are not provided, let user select them
    if corners is None:
        corners = select_corners(image)
    
    # Convert corners to numpy array
    corners = np.array(corners, dtype=np.float32)
    
    # Order corners in a consistent way (top-left, top-right, bottom-right, bottom-left)
    ordered_corners = order_points(corners)
    
    # Calculate the target rectangle dimensions for a top-down view
    # We want to maintain the aspect ratio that matches our desired grid
    rows, cols = grid_size
    width = cols * 100  # Arbitrary scale
    height = rows * 100  # Maintain square aspect ratio per cell
    
    # Define the destination points for the perspective transform (rectangle in top-down view)
    dst_pts = np.array([
        [0, 0],               # top-left
        [width, 0],           # top-right
        [width, height],      # bottom-right
        [0, height]           # bottom-left
    ], dtype=np.float32)
    
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
    
    # Generate grid points in the top-down view
    grid_points = []
    for row in range(rows + 1):
        for col in range(cols + 1):
            # Calculate normalized position (0 to 1)
            x = col / cols * width
            y = row / rows * height
            grid_points.append([x, y])
    
    grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Transform grid points back to the original image perspective
    M_inv = cv2.invert(M)[1]
    transformed_grid_points = cv2.perspectiveTransform(grid_points, M_inv)
    
    # Draw the grid on the image
    for i, point in enumerate(transformed_grid_points):
        x, y = point[0]
        cv2.circle(display_image, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # Draw grid lines
    # Horizontal lines
    for row in range(rows + 1):
        start_idx = row * (cols + 1)
        points = [tuple(map(int, transformed_grid_points[start_idx + col][0])) 
                 for col in range(cols + 1)]
        for i in range(len(points) - 1):
            cv2.line(display_image, points[i], points[i + 1], (0, 255, 0), 2)
    
    # Vertical lines
    for col in range(cols + 1):
        points = [tuple(map(int, transformed_grid_points[row * (cols + 1) + col][0])) 
                 for row in range(rows + 1)]
        for i in range(len(points) - 1):
            cv2.line(display_image, points[i], points[i + 1], (0, 255, 0), 2)
    
    # Draw the original rectangle
    for i in range(4):
        pt1 = tuple(map(int, ordered_corners[i]))
        pt2 = tuple(map(int, ordered_corners[(i + 1) % 4]))
        cv2.line(display_image, pt1, pt2, (255, 0, 0), 2)
    
    # Reshape grid points into a more useful format: (rows+1, cols+1, 2)
    grid_matrix = transformed_grid_points.reshape(rows + 1, cols + 1, 2)
    
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(display_image)
        plt.title(f'Isometric Rectangle with {rows}x{cols} Grid')
        plt.axis('off')
        plt.show()
    
    return display_image, grid_matrix

def order_points(pts):
    """
    Order points in: top-left, top-right, bottom-right, bottom-left order
    """
    # Sort by y-coordinate (top to bottom)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    
    # Get top two and bottom two points
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    
    # Sort top points by x (left to right)
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    
    # Sort bottom points by x (left to right)
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def select_corners(image):
    """
    Allow user to select 4 corners of the rectangle in the image
    """
    corners = []
    
    def on_click(event):
        if event.button == 1:  # Left mouse button
            corners.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()
            if len(corners) == 4:
                plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title('Click on the 4 corners of the rectangle (in any order)')
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    
    return corners

def get_cell_centers(grid_matrix):
    """
    Calculate the center point of each cell in the grid
    
    Parameters:
    - grid_matrix: Grid points matrix of shape (rows+1, cols+1, 2)
    
    Returns:
    - Array of cell centers of shape (rows, cols, 2)
    """
    rows, cols = grid_matrix.shape[0] - 1, grid_matrix.shape[1] - 1
    centers = np.zeros((rows, cols, 2))
    
    for r in range(rows):
        for c in range(cols):
            # Average the four corner points of each cell
            corners = [
                grid_matrix[r, c],
                grid_matrix[r, c+1],
                grid_matrix[r+1, c],
                grid_matrix[r+1, c+1]
            ]
            centers[r, c] = np.mean(corners, axis=0)
    
    return centers

def extract_cell_images(image, grid_matrix, cell_size=(100, 100)):
    """
    Extract images from each cell, corrected for perspective
    
    Parameters:
    - image: Original image
    - grid_matrix: Grid points matrix from divide_isometric_rectangle
    - cell_size: Size to make each extracted cell image
    
    Returns:
    - List of warped cell images
    """
    rows, cols = grid_matrix.shape[0] - 1, grid_matrix.shape[1] - 1
    cell_images = []
    
    for r in range(rows):
        row_images = []
        for c in range(cols):
            # Get the four corners of this cell
            src_points = np.array([
                grid_matrix[r, c],
                grid_matrix[r, c+1],
                grid_matrix[r+1, c+1],
                grid_matrix[r+1, c]
            ], dtype=np.float32)
            
            # Define destination points (square)
            dst_points = np.array([
                [0, 0],
                [cell_size[0], 0],
                [cell_size[0], cell_size[1]],
                [0, cell_size[1]]
            ], dtype=np.float32)
            
            # Calculate perspective transform
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transform
            warped = cv2.warpPerspective(image, M, cell_size)
            row_images.append(warped)
        
        cell_images.append(row_images)
    
    return cell_images

# Example usage
if __name__ == "__main__":
    # Path to your image with an isometric rectangle
    image_path = "Test/video_A_frame_0000.jpg"
    
    # Define the grid dimensions (rows, columns)
    grid_size = (5, 3)
    
    # You can either provide corners or select them manually
    # corners = [(100, 100), (300, 150), (250, 350), (50, 300)]  # Example corners
    
    # Divide the rectangle into a grid
    result_image, grid_matrix = divide_isometric_rectangle(
        image_path, 
        corners=None,  # Set to None to select corners manually
        grid_size=grid_size
    )
    
    # Get cell centers if needed
    centers = get_cell_centers(grid_matrix)
    
    # Extract individual cell images
    cell_images = extract_cell_images(cv2.imread(image_path), grid_matrix)
    
    # Display the first cell image as an example
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(cell_images[0][0], cv2.COLOR_BGR2RGB))
    plt.title("Example Cell Image")
    plt.axis('off')
    plt.show()