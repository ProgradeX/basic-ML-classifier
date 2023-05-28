import classifier
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

train_data = np.loadtxt("TrainData.csv")
train_labels = np.loadtxt("TrainLabels.csv")

# show images THIS DOSENT WORK ON .py FILES!!!
# plt.imshow(train_data[25].reshape([28, 28]))

method = 'hog'  # Choose method hog or edges
features = classifier.extract_features(train_data, method)

# Train the model using the complete training dataset
print("training RF model")
start = time.time()
rf_model = classifier.train_model(features, train_labels, 'random_forest')
rf_time = time.time()
print('RF training time: ', rf_time - start, ' seconds')

print("training SVM model")
start = time.time()
svm_model = classifier.train_model(features, train_labels, 'svm')
svm_time = time.time()
print('SVM training time: ', svm_time - start, ' seconds')


joblib.dump(rf_model, "rf_Model.pkl")
joblib.dump(svm_model, "svm_Model.pkl")

print('training complete :D')
