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

import sys
sys.path.append('..')
import visualization

# Convert depth and RGB images of the Siléane Dataset into a colored pointcloud
        
visualization.save_RGBD_to_colored_pointcloud('images/depth.PNG', 'images/rgb.PNG', 'images/camera_params.txt', "output_colored_pointcloud.ply", output_format="PLY")