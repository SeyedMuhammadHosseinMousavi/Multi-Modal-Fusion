%reset -f
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# Perform stratified train-test split for each modality
X_train_eye, X_test_eye, y_train_eye, y_test_eye = train_test_split(
    features_eye_standardized, labels_eye, test_size=0.2, random_state=42, stratify=labels_eye
)
X_train_gsr, X_test_gsr, y_train_gsr, y_test_gsr = train_test_split(
    features_gsr_standardized, labels_gsr, test_size=0.2, random_state=42, stratify=labels_gsr
)
X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = train_test_split(
    features_ecg_standardized, labels_ecg, test_size=0.2, random_state=42, stratify=labels_ecg
)

# Initialize classifiers for each modality
xgb_eye = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # XGBoost for EyeTracking
rf_gsr = RandomForestClassifier(random_state=42)  # RandomForest for GSR
xgb_ecg = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # XGBoost for ECG

# Train the classifiers on their respective modalities
print("Training classifiers for each modality...")
xgb_eye.fit(X_train_eye, y_train_eye)
rf_gsr.fit(X_train_gsr, y_train_gsr)
xgb_ecg.fit(X_train_ecg, y_train_ecg)

# Create a VotingClassifier for both hard and soft voting
voting_clf_hard = VotingClassifier(
    estimators=[('eye', xgb_eye), ('gsr', rf_gsr), ('ecg', xgb_ecg)], voting='hard'
)

voting_clf_soft = VotingClassifier(
    estimators=[('eye', xgb_eye), ('gsr', rf_gsr), ('ecg', xgb_ecg)], voting='soft'
)

# Train the voting classifiers
print("Training voting classifiers (hard and soft)...")
voting_clf_hard.fit(X_train_eye, y_train_eye)  # Use EyeTracking data to train VotingClassifier (this could be changed to combined features)
voting_clf_soft.fit(X_train_eye, y_train_eye)

# Predict using voting classifiers (hard and soft)
y_pred_hard = voting_clf_hard.predict(X_test_eye)  # You can change this to other test sets as needed
y_pred_soft = voting_clf_soft.predict(X_test_eye)

# Final classification report and confusion matrix for both voting methods
print("\nClassification Report (Hard Voting):")
print(classification_report(y_test_eye, y_pred_hard))

print("\nClassification Report (Soft Voting):")
print(classification_report(y_test_eye, y_pred_soft))

# Print confusion matrices
print("\nConfusion Matrix (Hard Voting):")
print(confusion_matrix(y_test_eye, y_pred_hard))

print("\nConfusion Matrix (Soft Voting):")
print(confusion_matrix(y_test_eye, y_pred_soft))

# Optionally, print accuracy scores
hard_voting_accuracy = accuracy_score(y_test_eye, y_pred_hard)
soft_voting_accuracy = accuracy_score(y_test_eye, y_pred_soft)

print(f"\nHard Voting Accuracy: {hard_voting_accuracy:.4f}")
print(f"Soft Voting Accuracy: {soft_voting_accuracy:.4f}")
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

# For Hard Voting
print("\nMetrics for Hard Voting:")
hard_voting_accuracy = accuracy_score(y_test_eye, y_pred_hard)
print(f"Test Accuracy (Hard Voting): {hard_voting_accuracy:.4f}")

# Calculate balanced accuracy
balanced_acc_hard = balanced_accuracy_score(y_test_eye, y_pred_hard)
print(f"Balanced Accuracy (Hard Voting): {balanced_acc_hard:.4f}")

# Calculate F1-Score (macro and weighted)
f1_macro_hard = f1_score(y_test_eye, y_pred_hard, average='macro')
f1_weighted_hard = f1_score(y_test_eye, y_pred_hard, average='weighted')
print(f"F1 Score (Macro) (Hard Voting): {f1_macro_hard:.4f}")
print(f"F1 Score (Weighted) (Hard Voting): {f1_weighted_hard:.4f}")

# Calculate Matthews Correlation Coefficient (MCC)
mcc_hard = matthews_corrcoef(y_test_eye, y_pred_hard)
print(f"Matthews Correlation Coefficient (MCC) (Hard Voting): {mcc_hard:.4f}")

# For Soft Voting
print("\nMetrics for Soft Voting:")
soft_voting_accuracy = accuracy_score(y_test_eye, y_pred_soft)
print(f"Test Accuracy (Soft Voting): {soft_voting_accuracy:.4f}")

# Calculate balanced accuracy
balanced_acc_soft = balanced_accuracy_score(y_test_eye, y_pred_soft)
print(f"Balanced Accuracy (Soft Voting): {balanced_acc_soft:.4f}")

# Calculate F1-Score (macro and weighted)
f1_macro_soft = f1_score(y_test_eye, y_pred_soft, average='macro')
f1_weighted_soft = f1_score(y_test_eye, y_pred_soft, average='weighted')
print(f"F1 Score (Macro) (Soft Voting): {f1_macro_soft:.4f}")
print(f"F1 Score (Weighted) (Soft Voting): {f1_weighted_soft:.4f}")

# Calculate Matthews Correlation Coefficient (MCC)
mcc_soft = matthews_corrcoef(y_test_eye, y_pred_soft)
print(f"Matthews Correlation Coefficient (MCC) (Soft Voting): {mcc_soft:.4f}")

# Concatenate the standardized features for all modalities into one
fused_data = pd.concat([features_eye_standardized, features_gsr_standardized, features_ecg_standardized], axis=1)
fused_data['Label'] = labels_eye  # Use the labels from EyeTracking (or any consistent label)

# Save the fused data to a CSV file
fused_data.to_csv('voting_late.csv', index=False)
print("Fused data saved to 'voting_late.csv'.")
