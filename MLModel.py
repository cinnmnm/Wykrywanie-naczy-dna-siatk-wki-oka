import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class MLModel:
    def __init__(self, model_type: str):
        if model_type == "kNN":
            self.model = KNeighborsClassifier()
        elif model_type == "drzewo":
            self.model = DecisionTreeClassifier()
        elif model_type == "las":
            self.model = RandomForestClassifier()
        elif model_type == "SVM":
            self.model = SVC()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test) -> np.ndarray:
        return self.model.predict(X_test)
