import pandas as pd
import numpy as np
from scipy.stats import mode
from Models.ensemble import EnsembleModel  

# 1. Load preprocessed train data
X_train = pd.read_csv("Preprocessed_data/X_train_preprocessed.csv")
y_train = pd.read_csv("Preprocessed_data/y_train.csv").squeeze()

# 2. Instantiate & train your ensemble
ensemble = EnsembleModel()
ensemble.train(X_train, y_train)

# 3. Load test data + PassengerId
X_test       = pd.read_csv("Preprocessed_data/X_test_preprocessed.csv")
passenger_ids = pd.read_csv("Preprocessed_data/test_passenger_ids.csv").squeeze()

# 4. Get each base‐model’s predictions via the wrapper’s .predict()
#    (uses your predict() forwarder in Model or calls into model.model)
preds = [ensemble.predict(X_test)]

# 5. Hard‐vote majority‐rule
ensemble_pred = mode(np.vstack(preds), axis=0, keepdims=False)[0].flatten().astype(int)

# 6. Build submission DataFrame
submission_df = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived":    ensemble_pred
})

# 7. Sanity checks
assert submission_df.shape[0] == len(passenger_ids)
assert list(submission_df.columns) == ["PassengerId", "Survived"]

# 8. Save CSV
submission_df.to_csv("submission.csv", index=False)
print("✓ submission.csv created")
