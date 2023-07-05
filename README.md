# NorthernBobwhiteCNN
https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.294

Code and data for training and evaluating a convolutional neural network (CNN) for Northern Bobwhite covey call detection.
Associated with 'The development of a convolutional neural network for the automatic detection of Northern Bobwhite Colinus virginianus covey calls' manuscript.

## Uploaded Data
We have uploaded a sample of our dataset of Northern Bobwhite covey call recordings in wav file format used to train the CNN: 90 4 second presence clips and 90 4 second absences clips (30 from each of the three groups of sites: WLFW-GA, WLFW-AL and DiLane). See: TrainingClips-Presence and TrainingClips-Absence folders for clips.

## Code Description
We have uploaded a trained bobwhite CNN, a .json file outlining the model parameters, the model source code and license. We also include a Jupyter Notebook demonstrating the trained model with an example covey call clip. The Jupyter Notebook outlines how to import the libraries, load the pre-trained model and clips and compute the model output. It also demonstrates plotting a mel-spectrogram on the clip, plotting the output predictions and obtaining the post-processing outputs of time and peak power of each detected covey call.

## NEW: Tutorial and File Structure
We have also now uploaded an example file structure for the model and data with all the necessary files, alongside a PDF containing instructions for how to re-run the model over your data and also how to retrain the model (including additional data augmentation) so that the model can be more focused on your specific data.
