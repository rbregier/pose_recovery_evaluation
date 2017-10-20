# Object detection and pose estimation toolbox

This repository provides tools to evaluate performance of an object detection and pose estimation method. It is based on the methodology introduced in the article [1] that is suited for any rigid object and can deal with scenes of many object instances, potentially occluded.

This toolbox has been primarily designed for use with the [Siléane Dataset](https://rbregier.github.io/dataset2017) but is generic and may be used with other datasets as well, provided with adapted ground truth annotations, and description of the object's geometry and symmetries (poseutils.json file).


## Current features
- evaluation_tools: computation of precision/recall curves with various evaluation goals, and various metrics (F1 score, Average Precision, Mean Average Precision, etc). 
- visualization: conversion of data from the Siléane Dataset into colored pointcloud.  
See the *examples* folder for typical use.


## Potential additional features

This toolbox is not complete yet.
Please feel free to contact me by email [2] or through GitHub for more information, feature requests, etc.
We want this repository to be useful for the community, and may provide additional tools is there is a demand.
Pull requests are welcome.


## Dependencies
Python 3+, Matplotlib, Numpy

## References
[1] Romain Brégier, Frédéric Devernay, Laetitia Leyrit and James L. Crowley, "Symmetry Aware Evaluation of 3D Object Detection and Pose Estimation in Scenes of Many Parts in Bulk", *in IEEE International Conference on Computer Vision Workshop (ICCVW), 2017.*

[2] Romain Brégier: contact information available at [rbregier.github.io](https://rbregier.github.io).