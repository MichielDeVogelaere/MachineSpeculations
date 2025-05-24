from sklearn.linear_model import LogisticRegression
from Models.model import Model


class LogisticRegressionModel(Model):

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)