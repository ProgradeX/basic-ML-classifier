import numpy as np
from skimage.feature import hog, canny
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def extract_features(images, method):
    features = []
    for image in images:
        if method == 'hog':
            resized_image = resize(image, (16, 16))
            hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            
            features.append(hog_features)
        
        elif method == 'edges':
            # Extract edges using Canny edge detection
            grayscale_image = rgb2gray(image)
            edges = canny(grayscale_image)
            
            edge_features = edges.flatten()
            features.append(edge_features)
            pass
    
    return np.array(features)


def train_model(features, labels, method):
    if method == 'random_forest':
        # Train a Random Forest classifier
        model = RandomForestClassifier()
        
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(features, labels)
        
        best_params = grid_search.best_params_
        print("Best hyperparameters for Random Forest:", best_params)
        model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
        model.fit(features, labels)
    
    elif method == 'svm':
        # Train Support Vector Machines classifier
        model = SVC()
        
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(features, labels)
        
        best_params = grid_search.best_params_
        print("Best hyperparameters for SVM:", best_params)
        model = SVC(C=best_params['C'], kernel=best_params['kernel'])
        model.fit(features, labels)
    
    return model