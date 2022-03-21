# NorthernBobwhiteCNN
Code and data for training and evaluating a convolutional neural network (CNN) for Northern Bobwhite covey call detection.

## Uploaded Data
We have uploaded a sample of our dataset of Northern Bobwhite covey call recordings in wav file format used to train the CNN: 90 4 second presence clips and 90 4 second absences clips (30 from each of the three groups of sites: WLFW-GA, WLFW-AL and DiLane). Due to the large size and confidential nature of the sites, all acoustic data is not uploaded here, but can be made available on request. See: TrainingClips-Presence and TrainingClips-Absence folders for clips.

## Code Description
We have uploaded a trained bobwhite CNN along with a .json file outlining the model parameters and model source code and license. We also include a Jupyter Notebook demonstrating the trained model with an example covey call clip. The Jupyter Notebook outlines how to import the libraries, load the pre-trained model and clips and compute the model output. It also demonstrates plotting a mel-spectrogram on the clip, plotting the output predictions and obtaining the post-processing outputs of time and peak power of each detected covey call.
