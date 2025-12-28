from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model
from Models.constants import SVCConstants

class SVCModel(Model):

    def __init__(self):
        super().__init__()
        self.model = SVC(
            probability=SVCConstants.PROBABILITY, 
            random_state=SVCConstants.RANDOM_STATE
        )

    def train(self, X_train, y_train):
        param_grid = {
            'C': SVCConstants.C_VALUES,
            'gamma': SVCConstants.GAMMA_VALUES
        }
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=SVCConstants.CV_FOLDS,
            scoring=SVCConstants.SCORING_METRIC,
            n_jobs=SVCConstants.N_JOBS,
            verbose=SVCConstants.VERBOSE
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

