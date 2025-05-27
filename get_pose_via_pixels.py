import numpy as np
import cv2

def pixel_to_world(u, v, depth, K, R_cw, t_cw):
    # Form the pixel in homogeneous coordinates
    pixel_h = np.array([u, v, 1.0])

    # Invert intrinsics
    K_inv = np.linalg.inv(K)

    # Backproject to camera frame
    point_cam = depth * (K_inv @ pixel_h)

    # Transform to world frame
    point_world = R_cw @ point_cam + t_cw
    return point_world

# Example inputs
u, v = 337, 297  # pixel coordinates

# points = [(168, 224), (420, 238), (154, 406), (420, 420)]
points = [(336, 126), (84, 126), (322, 308), (84, 294)]
# change x to y and y to x:
points = [(y, x) for x, y in points]

# [(336, 168), (112, 364), (350, 336), (98, 182)]

depth = 0.234                 # aruco depth from camera - box height (in meters)
# depth = 0.260 - 0.033                # aruco depth from camera - box height (in meters)

my_image = cv2.imread('target_heatmap_img.png')

for point in points:
    cv2.circle(my_image, point, radius=5, color=(0, 0, 255), thickness=-1)

cv2.imwrite("Marked Image.png", my_image)


# camera_matrix_old = np.array([[1161.8353084191926, 0.0, 343.5353683871253], 
#                       [0.0, 1147.038484476244, 208.88553172256437], 
#                       [0.0, 0.0, 1.0]])
    
# camera_matrix = np.array([[1141.268559300712, 0.0, 346.03487875496734], 
#                           [0.0, 1140.096576039464, 130.89923897039563], 
#                           [0.0, 0.0, 1.0]])
    
# adjusted on my own by looking at the depth -->
camera_matrix = np.array([[1036.268559300712, 0.0, 310.03487875496734], 
                      [0.0, 1036.096576039464, 210.89923897039563], 
                      [0.0, 0.0, 1.0]])
    
# camera_distortion_old = np.array([0.10610859066973946, 5.130084478704033, 7.463242263736177e-04, 0.006899619286465982, -44.639177399763526])

# camera_distortion = np.array([0.32894635553571083, -1.0523734016964843, -0.03135838933426853, 0.01917792608779873, 2.6164789509673287])

camera_distortion = np.array([0.18498355017606202, 1.5260447343986532, -0.001967072448315991, -0.0025235871527257356, -21.625173770519293])


R_cw = np.eye(3)  # Identity if camera is aligned with world
t_cw = np.zeros(3)  # No translation

point_3d = []

for point in points:
    u, v = point
    point_3d.append(pixel_to_world(u, v, depth, camera_matrix, R_cw, t_cw))
    
# Take mean of all points
point_3d_mean = np.mean(point_3d, axis=0)
print("3D Points:", point_3d)
print("Mean 3D world point:", point_3d_mean)

# breakpoint()

# 0.0078, 0.031, 0.2393