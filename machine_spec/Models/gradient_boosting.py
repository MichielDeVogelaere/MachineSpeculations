from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model

class GradientBoostingClassifierModel(Model):

    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier(random_state=42)

    def train(self, X_train, y_train):
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
