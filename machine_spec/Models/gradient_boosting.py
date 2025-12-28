from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model
from Models.constants import GradientBoostingConstants

class GradientBoostingClassifierModel(Model):

    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier()

    def train(self, X_train, y_train):
        param_grid = {
            'n_estimators': GradientBoostingConstants.N_ESTIMATORS,
            'learning_rate': GradientBoostingConstants.LEARNING_RATE,
            'max_depth': GradientBoostingConstants.MAX_DEPTH,
            'min_samples_split': GradientBoostingConstants.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': GradientBoostingConstants.MIN_SAMPLES_LEAF,
            'subsample': GradientBoostingConstants.SUBSAMPLE,  
            'max_features': GradientBoostingConstants.MAX_FEATURES
        }
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=GradientBoostingConstants.CV_FOLDS,
            scoring=GradientBoostingConstants.SCORING_METRIC,
            n_jobs=GradientBoostingConstants.N_JOBS,
            verbose=GradientBoostingConstants.VERBOSE
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
