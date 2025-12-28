from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model
from Models.constants import RandomForestConstants
import numpy as np

class RandomForestModel(Model):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(random_state=RandomForestConstants.RANDOM_STATE)
    
    def train(self, X_train, y_train):
        param_grid = {
            'n_estimators': RandomForestConstants.N_ESTIMATORS,
            'max_features': RandomForestConstants.MAX_FEATURES,
            'max_depth': RandomForestConstants.MAX_DEPTH,
            'min_samples_split': RandomForestConstants.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': RandomForestConstants.MIN_SAMPLES_LEAF,
            'bootstrap': RandomForestConstants.BOOTSTRAP
        }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,             
            cv=RandomForestConstants.CV_FOLDS,
            scoring=RandomForestConstants.SCORING_METRIC,
            n_jobs=RandomForestConstants.N_JOBS,
            verbose=RandomForestConstants.VERBOSE,
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
