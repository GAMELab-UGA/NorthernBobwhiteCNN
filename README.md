# NorthernBobwhiteCNN
https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.294

Code and data for training and evaluating a convolutional neural network (CNN) for Northern Bobwhite covey call detection.
Associated with 'The development of a convolutional neural network for the automatic detection of Northern Bobwhite Colinus virginianus covey calls'

## Uploaded Data
We have uploaded a sample of our dataset of Northern Bobwhite covey call recordings in wav file format used to train the CNN: 90 4 second presence clips and 90 4 second absences clips (30 from each of the three groups of sites: WLFW-GA, WLFW-AL and DiLane). Due to the large size and confidential nature of the sites, all acoustic data is not uploaded here, but can be made available on request. See: TrainingClips-Presence and TrainingClips-Absence folders for clips.

## Code Description
We have uploaded a trained bobwhite CNN, a .json file outlining the model parameters, the model source code and license. We also include a Jupyter Notebook demonstrating the trained model with an example covey call clip. The Jupyter Notebook outlines how to import the libraries, load the pre-trained model and clips and compute the model output. It also demonstrates plotting a mel-spectrogram on the clip, plotting the output predictions and obtaining the post-processing outputs of time and peak power of each detected covey call.

## NEW: Simple Tutorial
We have now added a simple tutorial that guides you through two of the main processes of our model:
  1) Running a pre-trained version of the CNN over your data
  2) Re-training the CNN using new data (including adding in new confusion species calls for data augmentation)
  
Download the CoveyCNN.zip file to access the tutorial, data and models!

This tutorial data also includes all of the data we used to train our original model for the manuscript.
