from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model
from Models.constants import KNNConstants

class KNeighborsClassifierModel(Model):

    def __init__(self):
        super().__init__()
        self.model = KNeighborsClassifier() # random state?

    def train(self, X_train, y_train):
        param_grid = {
            'n_neighbors': KNNConstants.N_NEIGHBORS,
            'weights': KNNConstants.WEIGHTS,
            'p': KNNConstants.P_VALUES  # Manhattan or Euclidean
        }
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=KNNConstants.CV_FOLDS,
            scoring=KNNConstants.SCORING_METRIC,
            n_jobs=KNNConstants.N_JOBS,
            verbose=KNNConstants.VERBOSE
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

