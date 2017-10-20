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



# In this file, we illustrate the evaluation of some results on a given dataset.


import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import evaluation_tools as etools
import poseutils



if __name__ == "__main__":
    # Ground truth folder
    ground_truth_foldername = "datasets/bunny_sample/gt"    
    # File describing object's geometry considered to estimate distance between poses
    poseutils_filename = "datasets/bunny_sample/poseutils.json"
    # Folder in which are stored results of object detection and pose estimation    
    results_foldername = "experiments/bunny_sample_results"    
    # Maximum occlusion rate of an instance to be considered of interest for pose recovery
    max_occlusion_rate = 0.5
    # Maximum number of results considered per scene, <0 means for no limitation. 
    max_nb_results = -1    
    
    ########        
    
    # Load utils class to deal with poses of this object
    poseut = poseutils.PoseUtils.load_from_file(poseutils_filename)
    
    # Load ground truth and results for each scene of the dataset
    scenes_data = etools.load_dataset_scenes_data(poseut, 
                                    ground_truth_foldername,
                                    results_foldername)
                                    
    # Estimate precision-recall curve for each scene, indexed by score, returned as a dictionnary                            
    pr_curves = etools.compute_precision_recall_curves_for_every_scene(scenes_data,
                   max_distance=poseut.distance_threshold, 
                   max_occlusion_rate = max_occlusion_rate, 
                   max_nb_results = max_nb_results, 
                   run_in_parallel = True)
                  
    # Estimate mean precision-recall curve
    # (Macro-averaged precision and recall values for decreasing scores values)
    mean_precisions, mean_recalls = etools.average_precision_recall_curves(list(pr_curves.values()))  
    plt.figure(1)
    plt.clf()            
    plt.step(mean_recalls, mean_precisions, where='post')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mean PR curve on the dataset")
    
    # Estimate some metrics
    metrics = {}
    metrics["F1"] = etools.get_best_F1_score(mean_precisions, mean_recalls)
    # Average Precision of the mean PR curve.
    metrics["AP"] = etools.get_average_precision(mean_precisions, mean_recalls)
    # Mean of Average Precision on each scenes.
    # These two metrics have slightly different meaning.
    metrics["MAP"] = etools.get_mean_average_precision(pr_curves.values())
    # Recall value at a given precision threshold (considering the mean PR curve)
    metrics["R99"] = etools.get_recall_at_precision(mean_precisions, mean_recalls, 0.99)
    metrics["R50"] = etools.get_recall_at_precision(mean_precisions, mean_recalls, 0.5)

   
    print("Performance metrics:")
    print("====================")
    print("Evaluation conditions:\n    max number of results per scene: {0}\n    Max occlusion rate of instances of interest: {1}%".format(max_nb_results, max_occlusion_rate * 100))
    for key, value in metrics.items():
        print("{0} : {1:.3f}".format(key, value))
    