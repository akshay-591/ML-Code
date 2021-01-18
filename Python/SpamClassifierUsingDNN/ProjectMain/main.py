"""
This is main File for Spam Classification model using Neural Network.
"""

import scipy.io as io
from DeepNetsOptimization import NNLearn, Prediction

# ======================== Training Using ANN ================
from ProjectMain import EmailProcessing

data_train = io.loadmat('../Data/spamTrain.mat')
X_train = data_train['X']
Y_train = data_train['y']


data_test = io.loadmat('../Data/spamTest.mat')
X_test = data_test['Xtest']
Y_test = data_test['ytest']

print("======================================= Data loaded ======================================")

Hidden_layer = 1
Hidden_layer_neuron = 10
input_layer_neurons = X_train.shape[1]  # input layer units
numClasses = 1  # output layer units

print("=========================================== Running Neural Network ============================================")
model = NNLearn.Learn(Input=X_train,
                      Output=Y_train,
                      AutoParameters=True,
                      maxIter=50,
                      InputLayerUnits=input_layer_neurons,
                      OutputLayerUnits=numClasses,
                      numHiddenLayers=Hidden_layer,
                      HiddenLayerUnits=Hidden_layer_neuron,
                      HiddenActivation="Sigmoid",
                      OutputActivation="Sigmoid",
                      lamb=5)

model = NNLearn.startTraining(model)

print("Accuracy on Training set = ", model.accuracy)

# Accuracy On Test set

prediction, accuracy = Prediction.predict(model, X_test, Y_test, Accuracy=True)
print("Accuracy on Test set = ", accuracy)

# ================================= Checking on real Emails =========================================
print("================================= Checking on real Emails =========================================")
email = open('../Data/spamSample1.txt', 'r+')
# process email and extracting feature vector
word_indices, feature_Vector = EmailProcessing.process(email)

predict = Prediction.predict(model, feature_Vector)

print("============================================ Result ===================================================")
if predict == 1:
    print("Email is Spam")
else:
    print("Email is not spam")
