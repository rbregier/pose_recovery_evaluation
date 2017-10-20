# Copyright 2017 Siléane
# Author: Romain Brégier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.image as mpimg
import poseutils
import numpy as np

def load_camera_params(filename):
    """ Load camera params and return them as a tuple (width, height, fu, fv, cu, cv, clip_start, clip_end, R, t)
    with (R,t) the pose of the camera expressed by a rotation matrix and a translation vector
    such as a point expressed in camera frame X_cam can be expressed in absolute coordinates system by X_world = np.dot(R, X_world) + t"""
    # Parse keys
    items = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            if len(line) > 0 and line[0] != '#':
                words = line.split()
                assert(len(words) > 1)
                if len(words) == 2:
                    items[words[0]] = float(words[1])
                else:
                    items[words[0]] = [float(v) for v in words[1:]]
                    
    R = poseutils.unit_quaternion_to_rotation_matrix([float(v) for v in items['rotation']])
    return (int(items["width"]), int(items["height"]),
         items["fu"], items["fv"],
         items["cu"], items["cv"], 
         items["clip_start"], items["clip_end"],
         R, items['location'])
                
def _float_to_uchar(x):
    return int(min(255, max(0, x * 255)))
              
           
def save_RGBD_to_colored_pointcloud(depth_img_filename, rgb_image_filename, camera_params_filename, output_filename, output_format = "PLY"):
    """ Transform input depth and RGB (or intensity) images into a colored point cloud expressed in absolute frame
    and saved in ASCII 'PLY' (default), or 'XYZ' format (X Y Z R G B).
    Output file can be opened in visualization softwares such as CloudCompare or MeshLab."""                   
    
    input_img=mpimg.imread(depth_img_filename)
    rgb_img = mpimg.imread(rgb_image_filename)
    assert(rgb_img.shape[0:2] == input_img.shape[0:2])
    # Cast intensity images to color images
    if len(rgb_img.shape) == 2:
        rgb_img = np.dstack((rgb_img, rgb_img, rgb_img))
    
    
    width, height, fu, fv, cu, cv, near, far, R, t = load_camera_params(camera_params_filename)
    assert(width==input_img.shape[1])
    assert(height==input_img.shape[0])
    
    depth_range = far - near
    assert(depth_range > 0)
    
    # Cast intensity values to depth values
    depth_map = near + depth_range * input_img
    
    # Compute 3D points coordinates, associated with color
    colored_points = []
    for v in range(height):
        for u in range(width):
            # Points with a white color in the input depth image are missing: we ignore them
            if input_img[v, u] < 1.0:              
                z_cam = depth_map[v, u]
           
           #if z_cam < far:      
                x_cam = z_cam / fu * (u - cu)
                y_cam = z_cam / fv * (v - cv)
               
                #X_world = np.dot(R, np.transpose([x_cam, y_cam, z_cam])) + t
                X_world = np.dot(R, [x_cam, y_cam, z_cam]) + t
                colored_points.append((X_world, rgb_img[v, u]))
    
    
    # Save to file for visualization in CloudCompare or other.
    if output_format == "XYZ":
        with open(output_filename, "w") as f:
            for point, color in colored_points:
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(point[0], point[1], point[2], color[0], color[1], color[2]))
    elif output_format == "PLY":
        with open(output_filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("comment Author: rbregier.github.com/pose_recovery_evaluation\n")
            f.write("element vertex {0}\n".format(len(colored_points)))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")   
            f.write("end_header\n")
            for point, color in colored_points:
                f.write("{x} {y} {z} {r} {g} {b}\n".format(
                    x = point[0],
                    y = point[1],
                    z = point[2],
                    r = _float_to_uchar(color[0]),
                    g = _float_to_uchar(color[1]),
                    b = _float_to_uchar(color[2])))
    else:
        raise ValueError("Invalid output format")