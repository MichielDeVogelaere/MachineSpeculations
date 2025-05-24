from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc, precision_score, recall_score, f1_score
)


class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_accuracy(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        accuracy = accuracy_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)
        return accuracy, roc_auc

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X) 

    def get_basic_metrics(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0)
        }
        
        # Only add ROC-AUC if model supports probability prediction
        if hasattr(self.model, 'predict_proba'):
            try:
                y_proba = self.model.predict_proba(X_val)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_val, y_proba)
            except:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
            
        return metrics