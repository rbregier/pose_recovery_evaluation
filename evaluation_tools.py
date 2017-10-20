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

import os
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import functools
import multiprocessing
import json

import sys


if os.name == 'nt':    
    path = os.path.dirname(__file__)
    # Without this I have troubles with multiprocessing on Windows
    if not (path in sys.path):
        sys.path.append(path)

# Various keys used to store information in dictionnaries
TRUTH_DATA = "truth_data"
RESULTS_DATA = "results_data"
NEAREST_TRUTH = "nearest_truth"
REPRESENTATIVES = "representatives"
OCCLUSION_RATE = "occlusion"
SCORE = "score"
TP="tp"
FP="fp"
FN="fn"

      
###########################
# BEGIN Base performance metrics
      
      
def get_F1_score(precision, recall):
    return 2*precision*recall/(precision + recall)

def get_best_F1_score(precisions, recalls, return_id = False):
    """Return the maximum F1 score achieved, and the associated uncertainety"""
    n = len(precisions)
    assert(n > 0)
    assert(len(recalls) == n)
    max_F1 = -1
    max_F1_id = -1
    for i in range(n):
        F1=get_F1_score(precisions[i], recalls[i])
        if F1 > max_F1:            
            max_F1 = F1
            max_F1_id = i
    if return_id:
        return max_F1, max_F1_id
    else:
        return max_F1        
          
def get_recall_at_precision(precisions, recalls, threshold, linear_interpolation=False):
    """ Compute and return the precision value obtained for a given threshold of recall.
    In case of multiple intersection of the precision-recall curve with the threshold,
    we consider the one with the highest recall value."""
    n = len(precisions)
    assert(n > 0)
    assert(len(recalls) == n)
    assert(threshold <= 1.0)
    
    # Find the recall of the last intersection of the threshold with the PR curve
    recall = -1.0      
    i = 0
    last_p = 1.0
    last_s = 0.0
    for i in range(n):
        p = precisions[i]
        s = recalls[i]
        if (p < threshold and last_p >= threshold):
            if linear_interpolation:
                # Linear interpolation of the precision-recall curve
                sen =(s * (last_p - threshold) + last_s * (threshold - p)) / (last_p - p)
            # No interpolation: PR curve is a step curve.
            sen = last_s
            if (sen > recall):            
                recall = sen
        last_p = p
        last_s = s
        
    if recall >= 0:
        return recall
    # There were no intersection -> the threshold is either always above or always below the curve.
    # But by definition the first PR curve value is [1, 0] and the precision threshold is below 1
    # therefore the precision is always above the threshold.
    # We return the highest recall value obtained.
    assert(precisions[n-1] >= threshold)
    return recalls[n-1]   

def is_sorted(x, key = lambda x: x): 
    return all([key(x[i]) <= key(x[i + 1]) for i in range(len(x) - 1)])

def get_average_precision(precisions, recalls):
    """ Compute and return the average precision, i.e. the integral under the precision/recall curve. """
    #sorted_p, sorted_r = _get_sorted_precisions_recalls_by_increasing_recall(precisions, recalls)
    # Recalls values should be always increasing
    assert(is_sorted(recalls))
    sorted_p = precisions
    sorted_r = recalls

    n = len(sorted_r)
    assert(len(sorted_p) == n)
    
    integral = 0
    last_r = 0
    for i in range(n):
        p = sorted_p[i]
        r = sorted_r[i]
        step = r - last_r
        assert(step >= 0)
        integral += p*step
        last_r = r
    return integral

def get_mean_average_precision(pr_curves_list):
    """ Return the Mean Average Precision of a set of experiments -- that is the mean of the Average Precisions obtained for each method. """
    """ Input: list of precision/recall curves indexed by score."""
    """ Note that this is different from the Average Precision of a mean precision-recall curve. """ 
    n = len(pr_curves_list)    
    assert(n > 0)
    sum_ap = 0
    for p, s, score in pr_curves_list:
        ap = get_average_precision(p, s)
        sum_ap += ap
    return sum_ap / n
  
# END metrics
########################
    


########################
# BEGIN results classification and precision/recall evaluation



def pose_data_to_dict(poseut, R, t):
    return {"R" : R, "t" : t, REPRESENTATIVES : poseut.get_representatives(R, t)}
    
def truth_data_to_dict(poseut, R, t, occlusion_rate):
    d = pose_data_to_dict(poseut, R, t)
    d[OCCLUSION_RATE] = occlusion_rate
    # occlusion_rate_uncertainety : ignored
    return d
    
def result_data_to_dict(poseut, R, t, score):
    d = pose_data_to_dict(poseut, R, t)
    d[SCORE] = score
    return d

        
def load_json_results_list(results_filename):
    """ Load results data expressed in a JSON file. Returns a list of dictionnaries having for keys: "R", "t", "score"""
    with open(results_filename, "r") as f:
        results_list = json.load(f)    
    return results_list

# Method that can be overloaded to load a different kind of results
_load_results_list = load_json_results_list
_results_file_extension = ".json"


def load_scene_data(poseut, ground_truth_filename : str, results_list):
    """ Load ground truth data, associated with some results. """
    scene_data = {TRUTH_DATA : [], RESULTS_DATA : [], NEAREST_TRUTH : []}
    
    #### Load ground truth
    with open(ground_truth_filename, "r") as f:
        gts = json.load(f)   
    # Assign data
    for gt in gts:
        scene_data[TRUTH_DATA].append( truth_data_to_dict(poseut, gt["R"], gt["t"],gt["occlusion_rate"]))

    ### assign results
    for results in results_list:
        scene_data[RESULTS_DATA].append(result_data_to_dict(poseut, results["R"], results["t"], results[SCORE]))   

    # Compute for each result the nearest ground truth instance
    if len(scene_data[TRUTH_DATA]) != 0:
        # Get representatives of every ground truth pose
        truth_reprs = []
        for t in scene_data[TRUTH_DATA]:
            truth_reprs.extend(t[REPRESENTATIVES])
        # Nearest neighbor search structure
        nneigh = NearestNeighbors(n_neighbors = 1,  algorithm = 'auto')
        nneigh.fit(truth_reprs)            
        # Estimate NN for each result and store it
        results = scene_data[RESULTS_DATA]
        scene_data[NEAREST_TRUTH] = [None] * len(results)
        for i in range(len(results)):
            r = results[i]
            dist, nn_id = nneigh.kneighbors([r[REPRESENTATIVES][0]])
            # Cast from numpy to basic type
            dist = dist[0][0]
            nn_id = nn_id[0][0]
            # Corresponding ground truth pose                
            truth_id = nn_id // poseut.nb_representatives
            scene_data[NEAREST_TRUTH][i] = (truth_id, dist)
    return scene_data



def load_dataset_scenes_data(poseut, ground_truth_foldername: str, results_foldername : str, decimation_factor = 1):
    """ Load ground truth results associated with each results from a dataset.\n
    Decimation factor :  consider only (1 out of decimation_factor) results for quick tests.
    Different file formats for results may be considered by reassigning the _load_results_list method and _results_file_extension variable. """
    scenes_data = {}
    
    names =  [name for name in os.listdir(results_foldername) if os.path.splitext(name)[1] == _results_file_extension]
    if decimation_factor > 1:
        names = names[::decimation_factor]
    for result_name in names:
        scene_name = os.path.splitext(result_name)[0]
        result_filename = os.path.join(results_foldername, result_name)
        ground_truth_filename = os.path.join(ground_truth_foldername, scene_name + ".json")
        assert(os.path.isfile(ground_truth_filename))
        results_list = _load_results_list(result_filename)
        scene_data = load_scene_data(poseut, ground_truth_filename, results_list)
        scenes_data[scene_name] = scene_data
    return scenes_data

def _classify(scene_data, max_distance, positives, to_find):
    """
    :param max_distance: Distance threshold for results validation
    :param positives: List of positives (list of ids)
    :param to_find: List of instances to find (list of ids)
    :return: classification
    """    
    c = {TP:[], FP:[], FN:[]}
    truth_data = scene_data[TRUTH_DATA]
    results_data = scene_data[RESULTS_DATA]  
    nearest_truth = scene_data[NEAREST_TRUTH]
    
    if len(positives) == 0:
        # No positives, thus all elements to find are false negatives
        c[FN] = to_find
    elif len(truth_data) == 0:
        # If there are no elements in the scene, all results are false positives
        c[FP] = positives
    else:
        # Get the list of representatives of every positive
        positives_representatives = []
        nb_representatives_per_pose = len(results_data[0][REPRESENTATIVES])    
        for p in positives:
            positives_representatives.extend(results_data[p][REPRESENTATIVES])    
        # For each ground truth pose to find, look for the nearest positive
        nearest_positive = {}
        nneigh = NearestNeighbors(n_neighbors = 1,  algorithm = 'auto')
        nneigh.fit(positives_representatives)   
        for t in to_find: # NOTE: ici je pourrais utiliser to_find à la place, sauf pour l'estimation des faux positifs
            dist, nn_id = nneigh.kneighbors([truth_data[t][REPRESENTATIVES][0]])
            # Cast from numpy to basic type
            dist = dist[0][0]
            nn_id = nn_id[0][0]
            # Corresponding positive id
            positive_id = nn_id // nb_representatives_per_pose
            nearest_positive[t] = positives[positive_id]
        # Classify positives, as true or false positives
        for p in positives:
            t, d = nearest_truth[p]
            if d > max_distance:
                # to far from a ground truth pose -> false positive
                c[FP].append(p)
            else:
                # Should the closest instance be found?
                is_to_find = (t in to_find)                
                
                # Check if p is a duplicate
                if is_to_find:
                    is_duplicate = (nearest_positive[t] != p)
                else:
                    # Look for the nearest positive to t (was not precomputed in nearest_positive)
                    dist, nn_id = nneigh.kneighbors([truth_data[t][REPRESENTATIVES][0]])
                    # Cast from numpy to basic type
                    dist = dist[0][0]
                    nn_id = nn_id[0][0]
                    # Corresponding positive id
                    nearest = positives[nn_id // nb_representatives_per_pose]
                    is_duplicate = (nearest != p) 
                
                # If it is a duplicate, it is considered as false positive
                if is_duplicate:
                    c[FP].append(p)
                elif is_to_find:
                    # else it is considered as true positive, if the corresponding ground truth pose was to find
                    c[TP].append((p, t))
                     
        # Classify false negatives
        c[FN] = [t for t in to_find for p in [nearest_positive[t]] if nearest_truth[p][0] != t or nearest_truth[p][1] > max_distance]      
    return c            


def classify_based_on_thresholds(scene_data, min_score, max_distance, max_occlusion_rate, min_occlusion_rate = -1.0, max_nb_results=-1):
    results_data = scene_data[RESULTS_DATA]
    truth_data = scene_data[TRUTH_DATA]
    # List of actual results given the score criterion
    
    positives = [i for i in range(len(results_data)) if results_data[i][SCORE] >= min_score]
    # Filter if a limited number of results is considered and if it is exceeded
    if (max_nb_results >= 0 and len(positives) > max_nb_results):
        # Sort by descending scores
        positives.sort(key=lambda i: results_data[i][SCORE], reverse = True)
        # Keep only the first max_nb_results positives
        positives = positives[0:max_nb_results]
    
    # List of ground truth instances to find given the occlusion rate criterion
    to_find = [i for i in range(len(truth_data)) if min_occlusion_rate <= truth_data[i][OCCLUSION_RATE] <= max_occlusion_rate]
    return _classify(scene_data, max_distance, positives, to_find)


def compute_recall(classification):
    ### Also called recall ###
    # Number of instances to be found
    t = len(classification[TP]) + len(classification[FN])
    if t == 0:
        return 1.0
    else:
        return len(classification[TP])/t


def compute_precision(classification):
    # Number of results
    r = len(classification[TP]) + len(classification[FP])
    if r == 0:
        return 1.0
    else:
        return len(classification[TP])/ r
        
def compute_recall_with_a_limited_number_of_retrieval(classification, max_nb_retrieval):
    ### Also called recall ###
    # Number of instances to be found
    tp = len(classification[TP])
    assert(tp <= max_nb_retrieval)
    t = min(max_nb_retrieval, tp + len(classification[FN]))
    if t == 0:
        return 1.0
    else:
        return tp/t    
        
# END results classification        
##########################
        
        
###########################
# BEGIN generation of precision/recall curves, indexed by decreasing score

def get_scores(scenes_data):
    """
    Return the set of scores attributed to each result of the scenes
    """
    return {r[SCORE] for scene_data in scenes_data.values() for r in scene_data[RESULTS_DATA] }
   

    
def compute_precision_recall_curve(max_distance, max_occlusion_rate, min_occlusion_rate, max_nb_results, scene_data):
    """ returns a tuple of precision/sensitivity lists, for every score, for the given scene """    
    pre = []
    sen = []
    
    scores = [math.inf] + list(get_scores({"toto" : scene_data}))
    scores = sorted(scores, reverse = True)
    for min_score in scores:
        c = classify_based_on_thresholds(scene_data, min_score, max_distance, max_occlusion_rate, min_occlusion_rate, max_nb_results)
        p = compute_precision(c)
        if max_nb_results <= 0:
            s = compute_recall(c)
        else:
            s = compute_recall_with_a_limited_number_of_retrieval(c, max_nb_results)
        pre.append(p)
        sen.append(s)
    return (pre, sen, scores)
    
def compute_precision_recall_curves_for_every_scene(
            scenes_data, 
            max_distance, 
            max_occlusion_rate = 1.1, 
            max_nb_results = -1, 
            run_in_parallel = False,
            min_occlusion_rate = -0.1):
    """ Return a dictionnary containing tuples of (precision/recall/associated score) lists for each scene. """
     
    mypartial = functools.partial(compute_precision_recall_curve, 
                                  max_distance, 
                                  max_occlusion_rate, 
                                  min_occlusion_rate, 
                                  max_nb_results
                                  )    
    if run_in_parallel:
        pool = multiprocessing.Pool(os.cpu_count()) 
        pre_sen_curves = pool.map(mypartial, scenes_data.values())
        pool.close()
    else:
        pre_sen_curves = []
        for scene_data in scenes_data.values():
            pre_sen_curves.append(mypartial(scene_data))
       
    # Store results
    res = {}
    i = 0
    for scene_name in scenes_data.keys():
        res[scene_name] = pre_sen_curves[i]
        i += 1
    return res

# END generation of precision/recall curves       
###########################        



def get_interpolated_precision_sensitivity(precisions, sensitivities):
    """ Return a step version of a precision/sensitivity curve with decreasing precision and increasing sensitivity"""
    # see fig. 8.2 of https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html for an example
    n = len(precisions)
    assert(len(sensitivities) == n)

    # Sort precision sensitivity data by decreasind sensitivity
    decreasing_presen = sorted([(precisions[i], sensitivities[i]) for i in range(n)], key = (lambda x : x[1]), reverse=True)
    
    inter_s = []
    inter_p = []
    pmax=0
    for p, s in decreasing_presen:
        pmax = max(p, pmax)
        inter_p.append(pmax)
        inter_s.append(s)
    inter_p.reverse()
    inter_s.reverse()
    return (inter_p, inter_s)
    
             
def average_precision_recall_curves(pr_curves, return_variance=False):
    """ Estimate macro averaged precision/recall curve based on a list of PR tuples for each score value.
    Assumption: PR curves are indexed by decreasing scores
    Returns a tuple containing the mean precision/recall/score data."""
    n_curves = len(pr_curves)
            
    # Lists indexed over the curves, of list indexed over scores
    precisions = []
    sensitivities = []
    for i in range(n_curves):
        precisions.append([])
        sensitivities.append([])
    scores = []
    current_score = math.inf
    ids = [0] * n_curves    
    while True:
        scores.append(current_score)
        # Greater score among the next ones after the results considered for each scene: it will be the next threshold value
        greatest_score = -math.inf
        for i, (p, r, s) in enumerate(pr_curves):
            # We look for the performances obtained with the last result having a score greater than the current score threshold
            while ids[i] < len(s) and  s[ids[i]] >= current_score:
                ids[i] += 1
            ids[i] -= 1
            cid = ids[i]
            assert(cid >= 0)
            assert(s[cid] >= current_score)                
            precisions[i].append(p[cid])
            sensitivities[i].append(r[cid])
            if not (len(precisions[i]) == len(scores)):
                print("bug")
            if cid + 1 < len(s):
                greatest_score = max(greatest_score, s[cid + 1])
        if greatest_score == -math.inf:
            break
        current_score = greatest_score
    
    # Compute statistical values, for each score
    mean_precisions = np.mean(precisions, axis = 0)
    mean_sensitivities = np.mean(sensitivities, axis = 0)
    if not return_variance:
        return (list(mean_precisions), list(mean_sensitivities))
    else:    
        var_precisions = np.var(precisions, axis = 0)
        var_sensitivities = np.var(sensitivities, axis = 0)
    
        return (list(mean_precisions), 
                list(mean_sensitivities),
                list(scores),
                list(var_precisions),
                list(var_sensitivities))        
          