from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from Models.model import Model


class LogisticRegressionModel(Model):

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    
    def train(self, X_train, y_train):
        param_grid = {
                'C': [0.001, 0.01, 0.1, 1],
                'penalty': ['l1', 'l2'], 
                'solver': ['liblinear'],
        }
            
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_ 