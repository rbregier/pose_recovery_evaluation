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

import json
import numpy as np
import math


class PoseUtils:
    def __init__(self, Rref2i, tref2i, distance_threshold = 0):
        self._Rref2i = np.array(Rref2i).copy()
        self._tref2i = np.array(tref2i).copy()
        self._distance_threshold = distance_threshold

    def _to_inertial_frame(self, R, t):
        """ Convert a pose from the base frame to its inertial frame """        
        return (self._Rref2i.dot(R), self._Rref2i.dot(t) + np.transpose(self._tref2i))
       
    def load_from_file(filename):
        f = open(filename, 'r')
        parsed = json.load(f)
        f.close()
        if 'type' in parsed:
            if parsed['type'] == AffinePoseUtils.__name__:
                return AffinePoseUtils(Lambda=parsed['Lambda'], 
                                           G=parsed['G'], 
                                            Rref2i = parsed['Rref2i'],
                                            tref2i = parsed['tref2i'],
                                            distance_threshold = parsed['distance_threshold'])
            elif parsed['type'] == RevolutionPoseUtils.__name__:
                return RevolutionPoseUtils(lambdaXZ = parsed['lambda'],
                                           rotoreflection_symmetry = parsed['rotoreflection_symmetry'],
                                            Rref2i = parsed['Rref2i'],
                                            tref2i = parsed['tref2i'],
                                            distance_threshold = parsed['distance_threshold'])

    def _to_dict(self):
        d = {}
        d['type'] = type(self).__name__
        d['Rref2i'] = self._Rref2i.tolist()
        d['tref2i'] = self._tref2i.tolist()
        d['distance_threshold'] = self._distance_threshold
        return d
        
    @property
    def distance_threshold(self):
        return self._distance_threshold

class AffinePoseUtils(PoseUtils):
    def __init__(self, Lambda, G, Rref2i, tref2i, distance_threshold=0):
        super().__init__(Rref2i, tref2i, distance_threshold)
        self._Lambda = np.array(Lambda).copy()
        #Proper symmetry group : list of 3x3 matrices
        self._G = [np.array(g).copy() for g in G]
        
    def save(self, filename):        
        data = self._to_dict()
        data['Lambda'] = self._Lambda.tolist()
        data['G'] = [g.tolist() for g in self._G]
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
 
    def get_representatives(self, R, t):
        """ returns representatives of the pose [R, t] """ 
        Ri, ti = self._to_inertial_frame(R, t)      
        reprs = []
        for g in self._G:
            #print(g)
            representative = np.concatenate( (np.reshape( np.dot(Ri, np.dot(g, self._Lambda)), 9, 'F' ), np.reshape(ti, 3)) )
            reprs.append(representative)
        return reprs
        
    @property
    def nb_representatives(self):
        return len(self._G)
        

class RevolutionPoseUtils(PoseUtils):
    def __init__(self, lambdaXZ, rotoreflection_symmetry, Rref2i, tref2i, distance_threshold=0):
        super().__init__(Rref2i, tref2i, distance_threshold)
        self._lambda = lambdaXZ
        self._rotoreflection_symmetry = rotoreflection_symmetry

    def save(self, filename):
        data = self._to_dict()
        data['lambda'] = self._lambda
        data['rotoreflection_symmetry'] = self._rotoreflection_symmetry
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
    def get_representatives(self, R, t):
        """ returns representatives of the pose [R, t] """ 
        Ri, ti = self._to_inertial_frame(R, t)      
        reprs = []
        representative = np.concatenate( (np.reshape(self._lambda * Ri[:,2], 3), np.reshape(ti, 3)) )
        reprs.append(representative)
        if self._rotoreflection_symmetry:
            representative = np.concatenate( (np.reshape(-self._lambda * Ri[:,2], 3), np.reshape(ti, 3)) )
            reprs.append(representative)
        return reprs
            
    @property
    def nb_representatives(self):
        if self._rotoreflection_symmetry:
            return 2
        else:
            return 1
    

def unit_quaternion_to_rotation_matrix(q):
    """ Convert from unit quaternion (w, x, y, z) to rotation matrix representation. """
    a, b, c, d = q
    # Reference (French): Wikipedia, "Quaternions et rotation dans l'espace", 2015/08/11
    # http://fr.wikipedia.org/wiki/Quaternions_et_rotation_dans_l'espace."
    return np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d], \
                        [2 * a * d + 2 * b * c, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b], \
                        [2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, a ** 2 - b ** 2 - c ** 2 + d ** 2]])
                           

def angle_axis_to_matrix(angle, axis):
    # Convert to quaternion
    cos = math.cos(angle/2)
    sin = math.sin(angle/2)
    # Scaled axe    
    sax = sin/np.linalg.norm(axis) * np.array(axis)
    q = [cos, sax[0], sax[1], sax[2]]
    return unit_quaternion_to_rotation_matrix(q)    
    
    
def Rz(theta):
    return angle_axis_to_matrix(theta, [0, 0, 1])
    
    
        
if __name__ == '__main__':
    print("poseutils test")
    
    
    if True:
        # Square root of the normalized covariance matrix
        Lambda = np.array([[24.798, 0.0, 0.0], [0.0, 24.798, 0.0], [0.0, 0.0, 67.9818]])
    
        #Proper symmetry group
        G = [Rz(0), Rz(math.pi/4), Rz(math.pi/2), Rz(3 * math.pi/4), Rz(math.pi), Rz(-3 * math.pi/4), Rz(-math.pi/2), Rz(-math.pi/4)]
        poseutils = AffinePoseUtils(Lambda, G, np.identity(3), np.zeros((3, 1)))
        assert(poseutils.nb_representatives == len(G))
    else:
        poseutils = RevolutionPoseUtils(0.1, True, np.identity(3), np.zeros((3, 1)))        
        
    # Test import/export
    poseutils.save("poseutils_test.temp")
    poseutils2 = PoseUtils.load_from_file("poseutils_test.temp")
    
    
    R1 = angle_axis_to_matrix(0.4, [1, 0, 1])
    t1 = np.transpose([-3, 0.2, 0])
    
    reprs = poseutils.get_representatives(R1, t1)
    reprs2 = poseutils2.get_representatives(R1, t1)


    assert(poseutils.nb_representatives == poseutils2.nb_representatives)
    
    assert(np.linalg.norm(np.array(reprs) - np.array(reprs2)) == 0)
        
    