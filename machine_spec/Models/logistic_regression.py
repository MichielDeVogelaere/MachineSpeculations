from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from Models.model import Model
from Models.constants import LogisticRegressionConstants


class LogisticRegressionModel(Model):

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(
            max_iter=LogisticRegressionConstants.MAX_ITER, 
            random_state=LogisticRegressionConstants.RANDOM_STATE
        )
    
    def train(self, X_train, y_train):
        param_grid = {
                'C': LogisticRegressionConstants.C_VALUES,
                'penalty': LogisticRegressionConstants.PENALTY_VALUES, 
                'solver': LogisticRegressionConstants.SOLVER_VALUES,
        }
            
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=LogisticRegressionConstants.CV_FOLDS,
            scoring=LogisticRegressionConstants.SCORING_METRIC,
            n_jobs=LogisticRegressionConstants.N_JOBS,
            verbose=LogisticRegressionConstants.VERBOSE,
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_ 