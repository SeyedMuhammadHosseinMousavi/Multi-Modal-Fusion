%reset -f
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

# Load the CSV files into pandas DataFrames
print("Loading data...")
eye_tracking_df = pd.read_csv('EyeTracking.csv')
gsr_df = pd.read_csv('GSR.csv')
ecg_df = pd.read_csv('ECG.csv')

# the first column is labeled consistently as 'Quad_Cat' for labels
print("Ensuring consistent column names...")
labels_eye = eye_tracking_df['Quad_Cat'].fillna(method='ffill')
labels_gsr = gsr_df['Quad_Cat'].fillna(method='ffill')
labels_ecg = ecg_df['Quad_Cat'].fillna(method='ffill')

# Drop the label column from the feature set
features_eye = eye_tracking_df.drop(columns=['Quad_Cat'])
features_gsr = gsr_df.drop(columns=['Quad_Cat'])
features_ecg = ecg_df.drop(columns=['Quad_Cat'])

# Padding DataFrames to the same number of rows (using NaN where data is missing)
max_rows = max(len(features_eye), len(features_gsr), len(features_ecg))

# Reindex each DataFrame to have the same number of rows
features_eye = features_eye.reindex(range(max_rows), fill_value=np.nan)
features_gsr = features_gsr.reindex(range(max_rows), fill_value=np.nan)
features_ecg = features_ecg.reindex(range(max_rows), fill_value=np.nan)

# Reindex the labels as well to match the length of the features
labels_eye = labels_eye.reindex(range(max_rows), fill_value=labels_eye.mode()[0])
labels_gsr = labels_gsr.reindex(range(max_rows), fill_value=labels_gsr.mode()[0])
labels_ecg = labels_ecg.reindex(range(max_rows), fill_value=labels_ecg.mode()[0])

# Impute missing values using mean strategy
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
features_eye_imputed = pd.DataFrame(imputer.fit_transform(features_eye), columns=features_eye.columns)
features_gsr_imputed = pd.DataFrame(imputer.fit_transform(features_gsr), columns=features_gsr.columns)
features_ecg_imputed = pd.DataFrame(imputer.fit_transform(features_ecg), columns=features_ecg.columns)

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
features_eye_standardized = pd.DataFrame(scaler.fit_transform(features_eye_imputed), columns=features_eye_imputed.columns)
features_gsr_standardized = pd.DataFrame(scaler.fit_transform(features_gsr_imputed), columns=features_gsr_imputed.columns)
features_ecg_standardized = pd.DataFrame(scaler.fit_transform(features_ecg_imputed), columns=features_ecg_imputed.columns)

# Combine all features from the three modalities into one dataset for the new classifier
print("Combining all modalities into one dataset for the new classifier...")
combined_features = pd.concat([features_eye_standardized, features_gsr_standardized, features_ecg_standardized], axis=1)

# Perform stratified train-test split for each modality and the combined dataset
X_train_eye, X_test_eye, y_train_eye, y_test_eye = train_test_split(
    features_eye_standardized, labels_eye, test_size=0.3, random_state=42, stratify=labels_eye
)
X_train_gsr, X_test_gsr, y_train_gsr, y_test_gsr = train_test_split(
    features_gsr_standardized, labels_gsr, test_size=0.3, random_state=42, stratify=labels_gsr
)
X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = train_test_split(
    features_ecg_standardized, labels_ecg, test_size=0.3, random_state=42, stratify=labels_ecg
)
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    combined_features, labels_eye, test_size=0.3, random_state=42, stratify=labels_eye
)

# Initialize classifiers for each modality and for the combined dataset
print("Initializing classifiers...")
xgb_eye = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # XGBoost for EyeTracking
rf_gsr = RandomForestClassifier(random_state=42)  # RandomForest for GSR
xgb_ecg = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # XGBoost for ECG
lr_combined = LogisticRegression(max_iter=1000)  # Logistic Regression for combined dataset

# Train the classifiers on each modality and the combined dataset
print("Training classifiers...")
xgb_eye.fit(X_train_eye, y_train_eye)
rf_gsr.fit(X_train_gsr, y_train_gsr)
xgb_ecg.fit(X_train_ecg, y_train_ecg)
lr_combined.fit(X_train_combined, y_train_combined)

# Predict the class probabilities for each modality and the combined dataset
print("Predicting class probabilities for test data...")
y_prob_eye = xgb_eye.predict_proba(X_test_eye)
y_prob_gsr = rf_gsr.predict_proba(X_test_gsr)
y_prob_ecg = xgb_ecg.predict_proba(X_test_ecg)
y_prob_combined = lr_combined.predict_proba(X_test_combined)

# Define the weights for each modality and the combined classifier (adjust as needed)
weights = [0.3, 0.2, 0.3, 0.2]  # Adjust these weights if needed

# Weighted averaging of the predicted probabilities from all classifiers
print("Performing weighted averaging of predictions...")
y_prob_weighted = (
    weights[0] * y_prob_eye +
    weights[1] * y_prob_gsr +
    weights[2] * y_prob_ecg +
    weights[3] * y_prob_combined
)

# Final prediction is the class with the highest averaged probability
y_pred_weighted = np.argmax(y_prob_weighted, axis=1)

# Final classification report and confusion matrix
print("\nClassification Report (Weighted Averaging):")
print(classification_report(y_test_eye, y_pred_weighted))

print("\nConfusion Matrix (Weighted Averaging):")
print(confusion_matrix(y_test_eye, y_pred_weighted))


# Print the final test accuracy
accuracy = accuracy_score(y_test_eye, y_pred_weighted)
print(f"\nTest Accuracy (Weighted Averaging): {accuracy:.4f}")
# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(y_test_eye, y_pred_weighted)
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
# Calculate F1-Score (macro and weighted)
f1_macro = f1_score(y_test_eye, y_pred_weighted, average='macro')
f1_weighted = f1_score(y_test_eye, y_pred_weighted, average='weighted')
print(f"F1 Score: {f1_macro:.4f}")
# Calculate Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test_eye, y_pred_weighted)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

# Save the final fused data (train + test) as a single CSV
fused_train_data = pd.concat([X_train_eye, X_train_gsr, X_train_ecg], axis=1)
fused_train_data['Label'] = y_train_eye  # Use the labels from EyeTracking (you can adjust this as needed)

fused_test_data = pd.concat([X_test_eye, X_test_gsr, X_test_ecg], axis=1)
fused_test_data['Label'] = y_test_eye  # Use the labels from EyeTracking (you can adjust this as needed)

# Combine the train and test data into one
fused_data = pd.concat([fused_train_data, fused_test_data])

# Save the fused data to a CSV file using the fusion method and time
fused_data.to_csv('weighted_avg_late.csv', index=False)
print("Fused data saved to 'weighted_avg_late.csv'.")
