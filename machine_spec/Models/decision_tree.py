from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from Models.model import Model

class DecisionTreeModel(Model):

    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(random_state=42)

    def train(self, X_train, y_train):

        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [5, 10, 20, 50],
            'min_samples_leaf': [2, 5, 10, 20],
            'criterion': ['gini', 'entropy'],
            'ccp_alpha': [0.0, 0.0001, 0.0005, 0.001]
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
