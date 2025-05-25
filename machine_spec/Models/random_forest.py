from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from Models.model import Model
import numpy as np

class RandomForestModel(Model):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(random_state=42)
    
    def train(self, X_train, y_train):
        param_distributions = {
            'n_estimators': np.arange(100, 301, 10),
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        randomized_search = RandomizedSearchCV(
            self.model,
            param_distributions=param_distributions,
            n_iter=100,              
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        
        randomized_search.fit(X_train, y_train)
        self.model = randomized_search.best_estimator_
