import numpy as np
import joblib

import classifier

test_data = np.loadtxt("TestData.csv")


method = 'hog'  # Choose method hog or edges
test_features = classifier.extract_features(test_data, method)

print('loading files...')
rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")
print('files loaded')

rf_predictions = rf_model.predict(test_features)
svm_predictions = svm_model.predict(test_features)

combined_predictions = list(zip(rf_predictions, svm_predictions))
np.savetxt('prediction.csv', combined_predictions, delimiter=',', fmt='%s')
print('csv file created!')

# i was high sorry
items = 0
mismatch = 0
for entry in combined_predictions:
    items += 1
    if entry[0] != entry[1]:
        mismatch += 1

print('both models agree on', 100 - ((mismatch * 100)/items), 'percent of data')
