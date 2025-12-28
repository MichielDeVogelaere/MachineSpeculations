from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model
from Models.constants import NaiveBayesConstants

class NaiveBayesModel(Model):
    def __init__(self):
        super().__init__()
        self.model = GaussianNB()
    
    def train(self, X_train, y_train):
        param_grid = {
            'var_smoothing': NaiveBayesConstants.VAR_SMOOTHING
        }
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=NaiveBayesConstants.CV_FOLDS,
            scoring=NaiveBayesConstants.SCORING_METRIC,
            n_jobs=NaiveBayesConstants.N_JOBS,
            verbose=NaiveBayesConstants.VERBOSE
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_