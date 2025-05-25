import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create the folder if it doesn't exist
os.makedirs("Preprocessed_data", exist_ok=True)

# 2. Load train and test data
train_df = pd.read_csv("OriginalData/train.csv")
test_df = pd.read_csv("OriginalData/test.csv")

# 3. Define target and features
target_column = "Survived"
X = train_df.drop(columns=[target_column])
y = train_df[target_column]

# Keep test PassengerId for submission later
test_passenger_ids = test_df["PassengerId"]

# 4. Handle missing values
X["Age"] = X["Age"].fillna(X["Age"].median())
test_df["Age"] = test_df["Age"].fillna(X["Age"].median())

X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
test_df["Embarked"] = test_df["Embarked"].fillna(X["Embarked"].mode()[0])

X["Fare"] = X["Fare"].fillna(X["Fare"].median())
test_df["Fare"] = test_df["Fare"].fillna(X["Fare"].median())

# 5. Drop columns with many missing/unique values + PassengerId
drop_cols = ["Ticket", "Name", "PassengerId"]
X.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# --- NEW FEATURE ENGINEERING ---

X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

X["hasCabin"] = X["Cabin"].notna().astype(int)
test_df["hasCabin"] = test_df["Cabin"].notna().astype(int)

X.drop(columns=["SibSp", "Parch", "Cabin"], inplace=True)
test_df.drop(columns=["SibSp", "Parch", "Cabin"], inplace=True)

# --- END NEW FEATURES ---

# 6. Encode categorical features
X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)
X_test = pd.get_dummies(test_df, columns=["Sex", "Embarked"], drop_first=True)

# Align columns in test with train (fill missing cols with 0)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 7. Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 8. Scale numerical features (add the new features to the list)
numerical_features = ["Age", "Pclass", "FamilySize", "Fare"]
scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


# 9. Save preprocessed data to disk in Preprocessed_data folder
X_train.to_csv("Preprocessed_data/X_train_preprocessed.csv", index=False)
X_val.to_csv("Preprocessed_data/X_val_preprocessed.csv", index=False)
y_train.to_csv("Preprocessed_data/y_train.csv", index=False)
y_val.to_csv("Preprocessed_data/y_val.csv", index=False)
X_test.to_csv("Preprocessed_data/X_test_preprocessed.csv", index=False)
test_passenger_ids.to_csv("Preprocessed_data/test_passenger_ids.csv", index=False)  # for submission
