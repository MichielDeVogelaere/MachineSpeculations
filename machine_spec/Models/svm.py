from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model

class SVCModel(Model):

    def __init__(self):
        super().__init__()
        self.model = SVC(probability=True, random_state=42)

    def train(self, X_train, y_train):
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
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

