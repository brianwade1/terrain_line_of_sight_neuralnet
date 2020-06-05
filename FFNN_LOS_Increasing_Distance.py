# FEED-FORWARD NEURAL NETWORK (FFNN) for Line-of-sight calculations in diffrent terrain.

#AUTHOR: BRIAN WADE, JOHN GRANT

# CALL-IN LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.externals import joblib
import sklearn.model_selection as model_selection
from keras.layers import Dense, Input, LocallyConnected1D, Dropout, BatchNormalization, Reshape, Flatten, LeakyReLU
from keras.models import Model, Sequential, load_model
from keras.constraints import maxnorm
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model

import keras
import os
import tensorflow as tf

# Save some models with pickle
import pickle
# !!! Saving models requires the h5py lib as well. !!!

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# import warnings filter and ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime
startTime = datetime.now()


# VARIABLES
training_size=.8 #the training, testing and validation sizes must equal 1 (Example: .8+.1+.1 =1.0)
validation_size=.5 #The validation set is a fraction of the test set.
np.random.seed(5)
epoch=200; #How many times to iterate over the training data.
#losses='sparse_categorical_crossentropy'
losses = 'binary_crossentropy'

metrics=['accuracy']
dropout1=0.2
alpha1=0.3

# Hyperparameters:
learning_rate = 0.01
batch1 = 2
HL1 = 250
HL2 = 250
HL3 = 500
HL4 = 100


#### Read and combine data  #####
#Data files
current_dir = os.getcwd()
data_folder = 'Data'
plains_data='plains.csv'
mountains_data='mountains.csv'
coast_data='coast.csv'

#Go to data folder and read in data into individual dataframes
#old_dir = os.getcwd()
#os.chdir('.\\data')
plains = pd.read_csv(os.path.join(current_dir, data_folder, plains_data), header=None)
mountains = pd.read_csv(os.path.join(current_dir, data_folder, mountains_data), header=None)
coast = pd.read_csv(os.path.join(current_dir, data_folder, coast_data), header=None)
#os.chdir(old_dir)

# combine data into a single dateframe
terrain_set = ['plains','mountains','coast']

for terrain in terrain_set:

    time_delta = datetime.now() - startTime
    delta_hour = time_delta.seconds//3600
    delta_min = ((time_delta.seconds - (delta_hour * 3600))//60)
    delta_sec = (time_delta.seconds - delta_hour*3600 - delta_min * 60)%60

    print('########################################################')
    print(f'Time elapsed since start of analysis: {delta_hour} hours, {delta_min} minutes, {delta_sec} seconds')
    print('starting terrain type: ' + terrain)
    print('########################################################')
    print(' ')

    # load current terrain set into a dataframe
    df = eval(terrain)

    #### Arrange data so that each LoS vector of 1000 pts is in a single row #####
    nrows = len(df) #Number of rows in the input file
    nsets = int(nrows/1000) #Divide the number of input file rows by 1000.
    input_set = np.empty((nsets, 1000))
    output_set = np.zeros((nsets, 1000))

    # Separate data into 1000 meter vectors
    ind = 0
    for k in range(nsets):
        end_ind = ind+1000
        input_chunk = df.iloc[ind:end_ind,2] #Use only the z-axis (altitude) data.
        output_chunk = df.iloc[ind:end_ind,4] #visibility - can see (0 or 1)
        input_set[k,:] = input_chunk  #use input_set for data, it is the correct shape. 
        output_set[k,:] = output_chunk #use output_set for output data.
        ind += 1000 
        
    #### SCALE DATA FROM 0 TO 1  #####
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(input_set)
    X = pd.DataFrame(data_scaled) #Fully-scaled data, range: 0 to 1

    # Save scaler info for later deployment
    scaler_filename = os.path.join(current_dir, 'models', 'FFNN_LOS_scaler.save')
    joblib.dump(min_max_scaler, scaler_filename) 

    # Note, the output_set has already essentially been scaled, so only scale the input_set.
    y = pd.DataFrame(output_set)

    #### Split data for training ####   
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=training_size, random_state=100, shuffle = True)
    X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, y_test, test_size=validation_size, random_state=1) 

    # initialize 
    end_pt_set = list(range(100,1100,100))
    score_set = np.zeros((len(end_pt_set), 4)) # This will hold the neural net fit R^2 scores [end_pt, train, val, test]
    index = 0

    # Loop through data with increasing end points
    for end_pt in end_pt_set:
    
        print('########################################################')
        print('starting end point: ' + str(end_pt) + ' for terrain type: ' + terrain)
        print('########################################################')
        
        X_train_now = X_train.iloc[:,0:end_pt]
        y_train_now = y_train.iloc[:,0:end_pt]

        X_val_now = X_val.iloc[:,0:end_pt]
        y_val_now = y_val.iloc[:,0:end_pt]

        X_test_now = X_test.iloc[:,0:end_pt]
        y_test_now = y_test.iloc[:,0:end_pt]

        ## NEURAL NETWORK TRAINING ##
        #### Prepare the network for training ####
        model = Sequential()
        input_dim = X_train_now.shape[1]

        # Input and hidden layers
        model.add(Dense(1000,input_dim=input_dim,activation='elu')) 
        model.add(Dropout(rate = 0.2))

        model.add(Dense(HL1))
        model.add(LeakyReLU(alpha=alpha1))
        model.add(Dropout(rate = dropout1)) 

        model.add(Dense(HL2, activation='relu'))
        model.add(Dropout(rate = dropout1))

        model.add(Dense(HL3, activation='relu'))
        model.add(Dropout(rate = dropout1))

        # model.add(Dense(HL4, activation='relu'))
        # model.add(Dropout(rate = dropout1))

        model.add(Dense(input_dim,activation='sigmoid'))
            
        #### Compile and train the network ####
        # optimizer1 = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # model.compile(optimizer=optimizer1, loss = losses, metrics = metrics)
        # model.fit(X_train, y_train, batch_size=batch1, epochs=epoch, validation_split=.1)

        # keras.optimizers.SGD(lr=0.99, momentum=0.99,  nesterov=True) 
        # model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
        # model.fit(X_train, y_train, epochs=epoch, batch_size=4, validation_split=.1)

        keras.optimizers.SGD(lr=0.99, momentum=0.99,  nesterov=True) 
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)


        model_name = os.path.join(current_dir, 'models', f'best_model_{terrain}_endpt_{end_pt}.h5')
        mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        model.fit(X_train_now, y_train_now, batch_size=1, epochs=epoch, validation_data=(X_val_now, y_val_now), callbacks=[es, mc])

        # load the saved model
        saved_model = load_model(model_name)    

        #### Evaluate the model fit ####
        scores1 = saved_model.evaluate(X_train_now, y_train_now, verbose=0)
        scores2 = saved_model.evaluate(X_val_now, y_val_now, verbose=0)
        scores3 = saved_model.evaluate(X_test_now, y_test_now, verbose=0)

        score_set[index,:] = [end_pt,scores1[1], scores2[1], scores3[1]]
        index += 1

        np.savetxt(os.path.join(current_dir, 'Results',f'results_{terrain}_increasing_dis.csv'), score_set, delimiter=',')


## Plot results
coast_results = np.array(pd.read_csv(os.path.join(current_dir, 'Results', 'results_coast_increasing_dis.csv'), header = None))
mountains_results = np.array(pd.read_csv(os.path.join(current_dir, 'Results', 'results_mountains_increasing_dis.csv'), header = None))
plains_results = np.array(pd.read_csv(os.path.join(current_dir, 'Results', 'results_plains_increasing_dis.csv'), header = None))

fig = plt.figure(figsize=(15, 5))
width = 0.2
ind = np.arange(len(coast_results))

ax = fig.add_subplot(131)
ax.bar(ind, plains_results[:,1], width, color = 'green', label = 'Plains')
ax.bar(ind + width, coast_results[:,1], width, color = 'blue', label = 'Coast')
ax.bar(ind + 2*width, mountains_results[:,1], width, color = 'grey', label = 'Mountains')
ax.set_xticks(ind + width)
ax.set_xticklabels((plains_results[:,0]).astype(int))
ax.set_xlabel('Predicted Distance (meters)')
ax.set_ylabel('Prediction Accuracy')
ax.set_title('Training Data')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)

ax = fig.add_subplot(132)
ax.bar(ind, plains_results[:,2], width, color = 'green', label = 'Plains')
ax.bar(ind + width, coast_results[:,2], width, color = 'blue', label = 'Coast')
ax.bar(ind + 2*width, mountains_results[:,2], width, color = 'grey', label = 'Mountains')
ax.set_xticks(ind + width)
ax.set_xticklabels((plains_results[:,0]).astype(int))
ax.set_xlabel('Predicted Distance (meters)')
#ax.set_ylabel('Prediction Accuracy')
ax.set_title('Validation Data')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)

ax = fig.add_subplot(133)
ax.bar(ind, plains_results[:,3], width, color = 'green', label = 'Plains')
ax.bar(ind + width, coast_results[:,3], width, color = 'blue', label = 'Coast')
ax.bar(ind + 2*width, mountains_results[:,3], width, color = 'grey', label = 'Mountains')
ax.set_xticks(ind + width)
ax.set_xticklabels((plains_results[:,0]).astype(int))
ax.set_xlabel('Predicted Distance (meters)')
#ax.set_ylabel('Prediction Accuracy')
ax.set_title('Test Data')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)

fig.tight_layout()
#plt.show()
plt.savefig(os.path.join(current_dir, 'Images', 'results_all_increasing_dis.png'))

# Save model summary
plot_model(model, to_file = os.path.join(current_dir, 'Images', 'NN_model.png'), show_shapes = True, show_layer_names = True)

## Calculate final run time and show complete
time_delta = datetime.now() - startTime
delta_hour = time_delta.seconds//3600
delta_min = ((time_delta.seconds - (delta_hour * 3600))//60)
delta_sec = (time_delta.seconds - delta_hour*3600 - delta_min * 60)%60

print(' ')
print(' ')
print('########################################################')
print('Computations Complete')
print(f'Time to complete analysis: {delta_hour} hours, {delta_min} minutes, {delta_sec} seconds')
print('########################################################')



