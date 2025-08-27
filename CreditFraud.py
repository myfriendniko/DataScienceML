import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ==============================================================================
# Simulate a synthetic dataset
# ==============================================================================
print("Generating synthetic dataset...")
n_samples = 100000
fraud_rate = 0.005 # 0.5% fraud
np.random.seed(42) # for reproducibility

# Create features
amounts = np.random.uniform(1, 2000, n_samples)
times = np.random.randint(0, 24*60*60, n_samples)
locations = np.random.randint(0, 50, n_samples) # 50 different locations

# Create a heavily imbalanced target variable (fraud)
is_fraud = np.random.choice([0, 1], n_samples, p=[1 - fraud_rate, fraud_rate])

# Introduce some fraud patterns (high amounts are more likely to be fraud)
high_amount_fraud_indices = np.random.choice(np.where(amounts > 1500)[0], int(n_samples * 0.002))
is_fraud[high_amount_fraud_indices] = 1

# Create the DataFrame
data = pd.DataFrame({
    'amount': amounts,
    'time': times,
    'location': locations,
    'is_fraud': is_fraud
})

print(f"Dataset generated. Total transactions: {len(data)}")
print(f"Fraudulent transactions: {data['is_fraud'].sum()}")
print(f"Legitimate transactions: {len(data) - data['is_fraud'].sum()}\n")

# ==============================================================================
# Data Preprocessing
# ==============================================================================
print("Starting data preprocessing...")

# Separate features (X) and target (y)
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Original training set shape: {X_train.shape}, {y_train.sum()} fraudulent cases")

# Apply SMOTE to the training data to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled training set shape: {X_train_resampled.shape}, {y_train_resampled.sum()} fraudulent cases\n")

# ==============================================================================
# Model Training
# ==============================================================================
print("Training the RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_resampled, y_train_resampled)
print("Model training complete.\n")

# ==============================================================================
# Model Evaluation
#
# I make predictions on the *original, imbalanced* test set
# to get a realistic performance metric.
# I use a classification report to view key metrics like Precision, Recall,
# and F1-score, which are more meaningful than simple accuracy for imbalanced data.
# ==============================================================================
print("Evaluating the model on the test set...")
y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# Explain the Confusion Matrix
print("\nConfusion Matrix values:")
print("True Negatives (TN): Top-left. Correctly predicted as legitimate.")
print("False Positives (FP): Top-right. Incorrectly predicted as fraud (a false alarm).")
print("False Negatives (FN): Bottom-left. Incorrectly predicted as legitimate (a missed fraud).")
print("True Positives (TP): Bottom-right. Correctly predicted as fraud.")

# Explain the importance of Recall
print("\n--- Key Metrics Explained ---")
print("For fraud detection, **Recall** for the '1' (fraud) class is often the most important metric.")
print("It tells us what percentage of all actual fraudulent transactions our model successfully caught.")
print("Our goal is to have a high Recall to minimize the number of missed fraud cases.")
print("The classification report shows our model achieved a Recall of {} for fraudulent cases.".format(round(classification_report(y_test, y_pred, output_dict=True)['1']['recall'], 2)))


