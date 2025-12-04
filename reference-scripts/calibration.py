import numpy as np
import cv2 as cv
import glob
import os
import tyro
from typing import Tuple

# Termination criteria for corner detection
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate_camera(
    images_dir: str = 'calibration_images',
    board_dim: Tuple[int, int] = (8, 6),
    square_size_mm: float = 30.0,
    visualize: bool = False,
):
    """
    Performs the camera calibration process.

    :param images_dir: Path to the folder containing calibration images.
    :param board_dim: Dimensions of the checkerboard's inner corners (rows, columns). Default (8, 6).
    :param square_size_mm: Actual side length of the checkerboard square (unit: mm). Default 30.0 mm.
    :param visualize: Whether to display images with detected corners for debugging.
    """
    print(f"--- Calibration Parameters ---")
    print(f"Image Directory: {images_dir}")
    print(f"Checkerboard Inner Corners (Rows, Cols): {board_dim}")
    print(f"Checkerboard Square Size: {square_size_mm} mm")

    # Arrays to store object points (3D) and image points (2D)
    world_points = []  # 3D points in real world space
    image_points = []  # 2D points in image plane

    # 1. Generate world coordinates for checkerboard corners
    # Assuming checkerboard is on the Z=0 plane, unit is square_size_mm
    total_corners = board_dim[0] * board_dim[1]
    world_coords_template = np.zeros((total_corners, 3), np.float32)
    world_coords_template[:, :2] = np.mgrid[0:board_dim[0], 0:board_dim[1]].T.reshape(-1, 2)
    world_coords_template = world_coords_template * square_size_mm

    # 2. Image path handling and processing
    if not os.path.exists(images_dir):
        print(f"ERROR: Image directory '{images_dir}' not found. Please create this folder and place calibration images inside.")
        return

    # Get all image paths
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    image_files.extend(glob.glob(os.path.join(images_dir, '*.png')))

    if not image_files:
        print(f"ERROR: No .jpg or .png images found in '{images_dir}'.")
        return

    print(f"\nFound {len(image_files)} images for corner detection...")
    
    # Used to get image dimensions
    first_gray_img = None 

    for fname in image_files:
        img = cv.imread(fname)
        if img is None:
            print(f"WARNING: Could not load image: {fname}")
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if first_gray_img is None:
            first_gray_img = gray # Save the first grayscale image to get dimensions

        # Find the checkerboard corners
        ret, corners = cv.findChessboardCorners(gray, board_dim, None)

        # If corners are found, refine them and record points
        if ret:
            # Add the template world points
            world_points.append(world_coords_template)
            
            # Refine corner positions to subpixel accuracy
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            image_points.append(corners2)
            
            print(f"Successfully found corners: {fname}")
            
            if visualize:
                # Draw and display corners
                img = cv.drawChessboardCorners(img, board_dim, corners2, ret)
                cv.imshow('Detected Corners', img)
                cv.waitKey(500)
        else:
            print(f"Could not find corners: {fname}")

    cv.destroyAllWindows()

    # 3. Perform Camera Calibration
    if len(image_points) == 0:
        print("\n❌ Failed to find enough corners in any image for calibration. Check checkerboard setup and image quality.")
        return

    # Dimensions of the grayscale image
    h, w = first_gray_img.shape[:2]

    print(f"\n--- Starting Camera Calibration using {len(image_points)} sets of data ---")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(world_points, image_points, (w, h), None, None)

    if ret:
        print("\n✅ Camera Calibration successful!")
        print("\n--- Calibration Results ---")
        
        # Camera Matrix (mtx)
        print("Camera Matrix (mtx):")
        print(mtx)
        
        # Distortion Coefficients (dist)
        # [k1, k2, p1, p2, k3, k4, k5, k6, ...]
        print("\nDistortion Coefficients (dist):")
        print(dist)
        
        # Calculate Mean Reprojection Error
        mean_error = 0
        for i in range(len(world_points)):
            imgpoints_reprojected, _ = cv.projectPoints(world_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(image_points[i], imgpoints_reprojected, cv.NORM_L2) / len(imgpoints_reprojected)
            mean_error += error
        
        print(f"\nMean Reprojection Error: {mean_error / len(world_points):.4f} pixels")
        
        # In practice, mtx and dist should be saved to a file
        # np.savez('camera_params.npz', mtx=mtx, dist=dist)
        
    else:
        print("\n❌ Camera Calibration failed. Check image quality and quantity.")

if __name__ == "__main__":
    # Use tyro.cli to parse command line arguments and pass them to the calibrate_camera function
    tyro.cli(calibrate_camera)