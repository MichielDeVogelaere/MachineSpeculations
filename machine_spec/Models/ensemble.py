from Models.logistic_regression import LogisticRegressionModel
from Models.random_forest import RandomForestModel
from Models.gradient_boosting import GradientBoostingClassifierModel 
from Models.naive_bayes import NaiveBayesModel
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import mode
import numpy as np
from Models.model import Model

class EnsembleModel(Model):

    def __init__(self):
        super().__init__()
        self.models = [
            LogisticRegressionModel(),
            RandomForestModel(),
            GradientBoostingClassifierModel(),
            NaiveBayesModel(),
        ]

    def train(self, X_train, y_train):
        for model in self.models:
            model.train(X_train, y_train)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        hard_voting_pred = mode(np.vstack(predictions), axis=0, keepdims=False)[0].flatten()
        return hard_voting_pred

    def predict_proba(self, X):
        probabilities = [model.predict_proba(X)[:, 1] for model in self.models]
        soft_voting_proba = np.mean(np.vstack(probabilities), axis=0)
        return np.vstack([1 - soft_voting_proba, soft_voting_proba]).T

    def get_accuracy(self, X_val, y_val):
        y_pred = self.predict(X_val)
        y_proba = self.predict_proba(X_val)[:, 1]
        accuracy = accuracy_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)
        return accuracy, roc_auc

    def get_basic_metrics(self, X_val, y_val):
        y_pred = self.predict(X_val)
        y_proba = self.predict_proba(X_val)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba)
        }

        return metrics
