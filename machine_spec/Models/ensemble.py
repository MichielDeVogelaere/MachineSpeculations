from Models.logistic_regression import LogisticRegressionModel
from Models.random_forest import RandomForestModel
from Models.gradient_boosting import GradientBoostingClassifierModel 
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import mode
import numpy as np
from Models.model import Model

class EnsembleModel(Model):

    def __init__(self):
        super().__init__()
        self.models = [
            # hardcoded models
            LogisticRegressionModel(),
            RandomForestModel(),
            GradientBoostingClassifierModel()
        ]

    def train(self, X_train, y_train):
       for model in self.models:
        model.train(X_train, y_train) 

    def get_accuracy(self, X_val, y_val):
        predictions = [model.predict(X_val) for model in self.models]
        hard_voting_pred = mode(np.vstack(predictions), axis=0, keepdims=False)[0].flatten()
        probabilities = [model.predict_proba(X_val)[:, 1] for model in self.models]
        soft_voting_proba = np.mean(np.vstack(probabilities), axis=0)

        accuracy = accuracy_score(y_val, hard_voting_pred)
        roc_auc = roc_auc_score(y_val, soft_voting_proba)

        return accuracy, roc_auc