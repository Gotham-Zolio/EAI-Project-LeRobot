import cv2
import numpy as np

def get_distorted_image(
    undistorted: np.ndarray, 
    W: int, 
    H: int, 
    mtx: np.ndarray, 
    dist: np.ndarray
) -> np.ndarray:
    """
    Simulates camera distortion by applying an inverse map to an undistorted image.
    This process finds where each pixel in the target distorted image should sample
    its color from the source undistorted image.

    :param undistorted: The source undistorted image (np.ndarray).
    :param W: Width of the image.
    :param H: Height of the image.
    :param mtx: Intrinsic camera matrix (K).
    :param dist: Distortion coefficients.
    :return: The resulting distorted image (np.ndarray).
    """
    image_size = (W, H)
    
    # 1. Calculate Optimal New Camera Matrix (P) for rectification (alpha=1.0)
    # This P matrix defines the coordinate system of the undistorted image.
    new_mtx_for_fov, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 1.0, image_size)
    
    # Manually set principal point (cx, cy) to the center of the image, 
    # ensuring the inverse mapping is centered.
    new_mtx_for_fov[0, 2] = W / 2
    new_mtx_for_fov[1, 2] = H / 2

    # 2. Generate coordinates for every pixel in the *target distorted* image (xd, yd)
    xs = np.arange(W)
    ys = np.arange(H)
    xx, yy = np.meshgrid(xs, ys)  # shape (H, W)

    # 3. Reshape coordinates into a format acceptable by cv2.undistortPoints: (N, 1, 2)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    points = points.reshape(-1, 1, 2)

    # 4. Apply undistortPoints to find source coordinates (mapx, mapy)
    # The output are the coordinates in the P camera space (undistorted_pts) 
    # that correspond to the distorted pixel coordinates.
    undistorted_pts = cv2.undistortPoints(points, cameraMatrix=mtx, distCoeffs=dist, R=None, P=new_mtx_for_fov)

    # 5. Reshape the map back to HxW format
    map_xy = undistorted_pts.reshape(H, W, 2)
    mapx = map_xy[:, :, 0].astype(np.float32)
    mapy = map_xy[:, :, 1].astype(np.float32)

    # 6. Remap: Apply the inverse mapping
    # The `remap` function samples the `undistorted` image at (mapx, mapy) coordinates
    # and places the result at the target pixel, effectively applying distortion.
    distorted = cv2.remap(undistorted, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return distorted


def simulate_distortion():
    # --- Camera Parameters ---
    W, H = 640, 480

    # Intrinsic Camera Matrix (mtx) - dtype=float64
    mtx = np.array([
        [570.21740069, 0., 327.45975405],
        [0., 570.1797441, 260.83642155],
        [0., 0., 1.]
    ], dtype=np.float64)

    # Distortion Coefficients (dist) - dtype=float64
    dist = np.array([
        -0.735413911,
        0.949258417,
        0.000189059234,
        -0.00200351391,
        -0.864150312
    ], dtype=np.float64)

    # Load the undistorted image (assumed to be from a pinhole model)
    input_filename = "camera_view.png"
    output_filename = "camera_view_distorted.png"
    
    undistorted = cv2.imread(input_filename)
    if undistorted is None:
        raise RuntimeError(f"Could not load {input_filename}. Please ensure the file exists.")

    # Call the core function
    distorted = get_distorted_image(undistorted, W, H, mtx, dist)

    cv2.imwrite(output_filename, distorted)
    print(f"Generated: {output_filename}")

if __name__ == "__main__":
    simulate_distortion()