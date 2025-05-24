from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model

class RandomForestModel(Model):

    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [4, 6, 8],
            'max_samples': [0.6, 0.8, 1.0]
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
