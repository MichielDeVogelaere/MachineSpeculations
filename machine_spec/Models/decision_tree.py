from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model
from Models.constants import DecisionTreeConstants

class DecisionTreeModel(Model):

    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(random_state=DecisionTreeConstants.RANDOM_STATE)

    def train(self, X_train, y_train):

        param_grid = {
            'max_depth': DecisionTreeConstants.MAX_DEPTH,
            'min_samples_split': DecisionTreeConstants.MIN_SAMPLES_SPLIT,
            'min_samples_leaf': DecisionTreeConstants.MIN_SAMPLES_LEAF,
            'criterion': DecisionTreeConstants.CRITERION,
            'ccp_alpha': DecisionTreeConstants.CCP_ALPHA
        }

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=DecisionTreeConstants.CV_FOLDS,
            scoring=DecisionTreeConstants.SCORING_METRIC,
            n_jobs=DecisionTreeConstants.N_JOBS,
            verbose=DecisionTreeConstants.VERBOSE
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_ 
