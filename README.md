# Fully Connected Feed-Forward Neural Network Predictions for line-of-sight Calculations

This program trains a fully connected feed-forward neural network to estimate if line-of-sight exists (binary, 0 or 1) for equally spaced points along a 2-dimensional terrain. The inputs to the model are the elevations of equally spaced points along the line-of-sight vector. The outputs are binary predictions of if line-of-sight exists between the observer and each point along the ground (the number nodes in the input and output layer are the same).

![Neural Network](/Images/NN_model_sample.png)

## Dataset

The data for this project was generated from 3D Elevation (3DEP) on the U.S. Geological Survey's (USGS) [National Geospatial Program](https://www.usgs.gov/core-science-systems/national-geospatial-program). It used [1/3 arc-second DEM](https://viewer.nationalmap.gov/basic/) data from three different regions of the United States: coastal, mountains, and plains. The data set consisted of random samples 2D slices of terrain 1,000 meters long and the associated line-of-sight calculations for each point relative to the observer. The data was sampled from three different regions:  

* Coastal areas: 1,048,576 samples of terrain
* Mountainous areas: 2,000,000 samples of terrain
* Plains areas: 2,000,000 samples of terrain

## Neural Net Model

The program used a fully-connected feed-forward neural network consisting of three hidden layers. The input layer and output layer have as many nodes as the length of the terrain vector that was being predicted and used an [exponential  linear activation function](https://keras.io/api/layers/activations/) (elu). The first hidden layer used 250 hidden nodes with a [leaky rectified linear activation function](https://keras.io/api/layers/activations/) (LRelu). The second hidden layer also had 250 hidden nodes with a rectified linear (Relu) activation function. The third hidden layer had 500 nodes also with a Relu activation function. Each hidden layer was followed by a dropout layer with a 20% probability in order to increase its generalizability. Finally, the output layer used a [sigmoid](https://keras.io/api/layers/activations/) activation function. During the training process, the scaler used to normalize the data and the best neural network for each terrain set and prediction length are stored to the models folder.

![Sample Output](/Images/NN_model.png)

## Results

The program calculates the accuracy of the trained model on predicting line-of-sight for distances from 100 meters to 1,000 meters. The model was trained with the training data (80% of the full dataset), a validation set (10% of the full dataset) was used for early stopping during training increase in the binary cross entropy loss function for the validation set with 10 epochs patience), and the models accuracy tested with a test set (10% of the full dataset withheld from all training). The results show the intuitive results that the accuracy for predicting line-of-sight was greatest for plains-type of terrain and smallest for mountain-type terrain. Additionally, the accuracy generally decreased for predictions of longer sets (more distant) terrain.

![Sample Output](/Images/results_all_increasing_dis.png)

## Setup and Run

The program loops through the three train sets and tests ten different distance measures from 0-100 meter out to 0-1000 meters in 100 meter increments. As a result, *it takes about 10 hours to run* on a standard home computer. The simulation is setup, run, and controlled from the [FFNN_LOS_Increasing_Distance.py](FFNN_LOS_Increasing_Distance.py) file. The simulation parameters are set in lines 40-57. The size of the training set is controlled by "training_size" and "validation_size." The "validation_size" is the percentage of the data left over after the training set is removed that should be allocated to the validation set. The maximum allowed epochs during the training of any dataset is controlled by the "epoch" variable. Note that the training may end before this epoch due to early stopping. The number of hidden nodes in each hidden layer (1-3) is controlled with "HL1," "HL2," and "HL3." The dropout at percentage (percent of each connections randomly removed from training at each epoch) is controlled with "dropout" and the slop of the negative portion of the leaky Relu is set with the "alpha1" variable. Finally, the learning rate of the [adam optimizer](https://keras.io/api/optimizers/) is set with the "learning_rate" variable.

![Inputs](/Images/Inputs.png)

## Credits

This simulation was built with the help of the following people:

* Jim Jablonski
* John Grant

## Python Libraries

This simulation was built using the following python libraries and versions:

* Python v3.7.6
* numpy v1.18.1
* matplotlib v3.1.3
* tensorflow v1.15.0
* keras v2.2.4
* scikit-learn v0.22.1
* pickpleshare v0.7.5
* h5py v2.10.0
* graphviz v2.38
* pydot v1.4.1
