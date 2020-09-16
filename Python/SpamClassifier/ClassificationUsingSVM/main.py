# this file contains code for Spam Classifier

import numpy as mat
import scipy.io as io
from ClassificationUsingSVM import EmailProcessing, VocabArray, SMO, Prediction


email = open('emailSample1.txt', 'r+')
# process email and extracting feature vector
word_indices, feature_Vector = EmailProcessing.process(email)

print("=========================== Extracting features =================")
print("length of feature vector = ", len(VocabArray.getVocab()))
print("number of non zero are  =  ", len(mat.where(feature_Vector == 1)[0]))

# ======================== Training Using SVM ================
data = io.loadmat('spamTrain.mat')
X = data['X']
Y = data['y']
print("data loaded")
print(X.shape)
# Replacing 0 with -1 in Y dataset for ease of calculation
y = mat.where(Y == 0, -1, Y)
C = 0.1
# vectorised dataset
<<<<<<< HEAD
# since our example set is big in Test set of (4000,1849) in python it take longer time for dot product
# to save time we did the dot product in octave and saved it in .mat file

# load kernel dataset
kernel = io.loadmat('kernel.mat')
k = kernel['k']

smo_model = SMO.simplifiedSMO(X, y, C, k)
r = SMO.execute_SMO(smo_model, 5)
p = Prediction.predict(smo_model, X, "linear")
=======
k = SMO.linear_kernel(X)
smo_model = SMO.simplifiedSMO(X, y, C, k)
r = SMO.execute_SMO(smo_model, 5)
p = Prediction.predict(smo_model, X, "linear")
accuracy = mat.subtract(p, Y)
accuracy = len(mat.where(accuracy == 0)[0])
accuracy = (accuracy/len(Y))*100
print("Accuracy using SMO = ",accuracy)
print("================ Testing Original SMO ======================")
model = OriginalSMO.SMO(X, y, C, k)
new_model = OriginalSMO.execute(model)
print("done")
p = Prediction.predict(new_model, X, "linear")
>>>>>>> parent of 9e8a657... Original SMO updated -- objective Function
accuracy = mat.subtract(p, Y)
accuracy = len(mat.where(accuracy == 0)[0])
accuracy = (accuracy / len(Y)) * 100
print("Training set Accuracy  = ", accuracy)

# ================= Testing on Test set ==============
test_set = io.loadmat('spamTest.mat')
test_X = test_set['Xtest']
test_Y = test_set['ytest']
y = mat.where(Y == 0, -1, Y)
p = Prediction.predict(smo_model, test_X, "linear")
accuracy = mat.subtract(p, test_Y)
accuracy = len(mat.where(accuracy == 0)[0])
accuracy = (accuracy / len(test_Y)) * 100
print("Test set Accuracy  = ", accuracy)

# ================= Top Predictors of Spam ==================== Since the model we are training is a linear SVM,
# we can inspect the weights learned by the model to understand better how it is determining whether an email is spam
# or not. The following code finds the words with the highest weights in the classifier. Informally,
# the classifier 'thinks' that these words are the most likely indicators of spam.

weights = smo_model.W
sorted_weights = mat.sort(weights,axis=0)[::-1]
vocab = VocabArray.getVocab()
for i in range (15):

    j = mat.where(weights == sorted_weights[i])[0]
    print(vocab[j], "    ", sorted_weights[i])

# ======================= Testing on email =================
email = open('spamSample1.txt', 'r+')
# process email and extracting feature vector
word_indices, feature_Vector = EmailProcessing.process(email)

pred = Prediction.predict(smo_model, feature_Vector, 'linear')
print("====================== Result ====================")
if pred == 1:
    print("email is spam")
else:
    print("email is not spam")