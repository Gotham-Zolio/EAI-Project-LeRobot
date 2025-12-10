import cv2
import numpy as np

# Front camera intrinsics
FRONT_CAM_W = 640
FRONT_CAM_H = 480
FRONT_FX = 570.21740069
FRONT_FY = 570.17974410
FRONT_CX = FRONT_CAM_W / 2
FRONT_CY = FRONT_CAM_H / 2

# Distortion coefficients (for applying to front camera image)
K1 = -0.735413911
K2 = 0.949258417
P1 = 0.000189059
P2 = -0.002003513
K3 = -0.864150312

def apply_distortion(img, fx, fy, cx, cy, k1=K1, k2=K2, p1=P1, p2=P2, k3=K3):
    """
    Apply radial + tangential distortion to an image (numpy array HxWxC uint8).
    fx,fy,cx,cy should match the camera intrinsics used to render the image.
    """
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xd = (xs - cx) / fx
    yd = (ys - cy) / fy
    r2 = xd * xd + yd * yd
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x = (xd - 2 * p1 * xd * yd - p2 * (r2 + 2 * xd * xd)) / radial
    y = (yd - p1 * (r2 + 2 * yd * yd) - 2 * p2 * xd * yd) / radial
    u = (x * fx + cx).astype(np.float32)
    v = (y * fy + cy).astype(np.float32)

    distorted_img = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return distorted_img
