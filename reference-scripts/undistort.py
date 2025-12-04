import cv2
import numpy as np
import math
import tyro
import os
from typing import Tuple

# --- Calibration Constants ---
# Hardcoded for demonstration purposes.

# Image resolution (W, H)
W, H = 640, 480
IMAGE_SIZE = (W, H)

# Intrinsic Camera Matrix (mtx)
MTX = np.array([
    [570.21740069,   0.,             327.45975405],
    [0.,             570.1797441,    260.83642155],
    [0.,             0.,             1.          ]
], dtype=np.float32)

# Distortion Coefficients (dist) [k1, k2, p1, p2, k3]
DIST = np.array([
    -0.735413911,
    0.949258417,
    0.000189059234,
    -0.00200351391,
    -0.864150312
], dtype=np.float32)


def undistort_and_analyze_image(
    image_path: str = 'front_camera.png',
    output_dir: str = 'undistorted_results'
):
    """
    Loads an image, performs undistortion using predefined calibration parameters,
    analyzes the camera's Field of View (FOV), and saves the results.

    :param image_path: Path to the distorted image to be processed.
    :param output_dir: Directory where the undistorted images will be saved.
    """
    print("--- Loaded Exact Calibration Parameters ---")
    print(f"Resolution: {W} x {H}")
    print(f"Camera Matrix (MTX):\n{MTX.round(4)}")
    print(f"Distortion Coefficients (DIST): {DIST.round(6)}")
    print("------------------------------------------")

    # --- 1. FOV Analysis ---
    
    # Compute optimal new camera matrix (alpha=1.0 maximizes view)
    new_mtx_for_fov, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, IMAGE_SIZE, 1.0, IMAGE_SIZE)

    # Extract new focal lengths
    new_fx = new_mtx_for_fov[0, 0]
    new_fy = new_mtx_for_fov[1, 1]

    # Calculate rectified (effective) horizontal and vertical FOV (in radians)
    new_fov_x_rad = 2 * np.arctan(W / (2 * new_fx))
    new_fov_y_rad = 2 * np.arctan(H / (2 * new_fy))

    # Convert to degrees
    new_fov_x_deg = np.degrees(new_fov_x_rad)
    new_fov_y_deg = np.degrees(new_fov_y_rad)

    # Calculate original (ideal, pre-distortion) FOV
    orig_fov_x_rad = 2 * np.arctan(W / (2 * MTX[0, 0]))
    orig_fov_y_rad = 2 * np.arctan(H / (2 * MTX[1, 1]))
    orig_fov_x_deg = np.degrees(orig_fov_x_rad)
    orig_fov_y_deg = np.degrees(orig_fov_y_rad)
    
    print(f"\n--- Field of View (FOV) Analysis ---")
    print(f"Original (Ideal) FOV: {orig_fov_x_deg:.2f}° (H) x {orig_fov_y_deg:.2f}° (V)")
    print(f"Rectified (Effective) FOV: {new_fov_x_deg:.2f}° (H) x {new_fov_y_deg:.2f}° (V)")
    print("----------------------------------------")


    # --- 2. Image Undistortion Pipeline ---

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 2.1 Read Image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image '{image_path}'. Creating a placeholder image.")
            img = np.zeros((H, W, 3), dtype=np.uint8) 
            cv2.putText(img, "Placeholder Image (File Not Found)", (W//6, H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # 2.2 Optimize New Camera Matrix and get Region of Interest (ROI)
        # alpha=1.0 maximizes the view
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(MTX, DIST, IMAGE_SIZE, 1, IMAGE_SIZE)

        # 2.3 Undistort using cv2.remap()
        # 2.3a Calculate the mapping tables
        mapx, mapy = cv2.initUndistortRectifyMap(MTX, DIST, None, new_mtx, IMAGE_SIZE, cv2.CV_32FC1) 

        # 2.3b Remap the image pixels
        dst_remap = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # 2.4 Crop Image
        x, y, w_crop, h_crop = roi
        # Only crop if the ROI is valid
        if w_crop > 0 and h_crop > 0:
            dst_cropped = dst_remap[y:y+h_crop, x:x+w_crop]
        else:
            dst_cropped = dst_remap
            print("WARNING: ROI for cropping is invalid. Skipping crop.")


        # 2.5 Save Results
        undistorted_path = os.path.join(output_dir, 'undistorted_image.png')
        cropped_path = os.path.join(output_dir, 'undistorted_image_cropped.png')

        cv2.imwrite(undistorted_path, dst_remap)
        cv2.imwrite(cropped_path, dst_cropped)
        
        print("\n--- Processing Results ---")
        print("Image successfully undistorted.")
        print(f"Original image size: {img.shape}")
        print(f"Undistorted image size: {dst_remap.shape}")
        print(f"Cropped image size: {dst_cropped.shape}")
        print(f"Saved undistorted image to: {undistorted_path}")
        print(f"Saved cropped undistorted image to: {cropped_path}")


    except Exception as e:
        print(f"ERROR during image processing: {e}")

if __name__ == "__main__":
    # Use tyro.cli to parse command line arguments
    tyro.cli(undistort_and_analyze_image)