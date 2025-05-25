from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model

class GradientBoostingClassifierModel(Model):

    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier()

    def train(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 20],
            'min_samples_leaf': [1, 10],
            'subsample': [0.8, 0.9, 1.0],  
            'max_features': ['sqrt', 'log2', None]
        }
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
