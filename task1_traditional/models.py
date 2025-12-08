import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class MulticlassSVM:
    
    def __init__(self):
        self.classifiers = []
        self.scaler = StandardScaler()
        self.classes = []

    def train(self, X, y):

        X_scaled = self.scaler.fit_transform(X)
        self.classes = np.unique(y)
        self.classifiers = []
        
        print(f"Training {len(self.classes)} binary SVMs...")
        
        for i, cls in enumerate(self.classes):
            y_binary = np.where(y == cls, 1, -1)
            clf = SVC(kernel='linear', random_state=42, max_iter=10000)
            clf.fit(X_scaled, y_binary)
            self.classifiers.append(clf)
            
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        scores = np.zeros((n_samples, n_classes))

        for i, clf in enumerate(self.classifiers):
            scores[:, i] = clf.decision_function(X_scaled)

        pred_indices = np.argmax(scores, axis=1)
        return self.classes[pred_indices]

class TraditionalClassifier:
    def __init__(self):
        self.model = MulticlassSVM()

    def train(self, X, y):
        self.model.train(X, y)

    def predict(self, X):
        return self.model.predict(X)
