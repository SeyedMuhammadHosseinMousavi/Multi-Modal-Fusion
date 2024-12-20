%reset -f
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the CSV files into pandas DataFrames
ecg_df = pd.read_csv('ECG.csv')
eye_tracking_df = pd.read_csv('EyeTracking.csv')
gsr_df = pd.read_csv('GSR.csv')

# the first column is labeled consistently as 'Label'
ecg_df.columns = ['Label'] + list(ecg_df.columns[1:])
eye_tracking_df.columns = ['Label'] + list(eye_tracking_df.columns[1:])
gsr_df.columns = ['Label'] + list(gsr_df.columns[1:])

# Adding prefixes to feature columns to have unique feature names
ecg_df.columns = ['Label'] + ['ECG_' + col for col in ecg_df.columns[1:]]
eye_tracking_df.columns = ['Label'] + ['Eye_' + col for col in eye_tracking_df.columns[1:]]
gsr_df.columns = ['Label'] + ['GSR_' + col for col in gsr_df.columns[1:]]

# Padding DataFrames to the same number of rows (using NaN where data is missing)
max_rows = max(len(ecg_df), len(eye_tracking_df), len(gsr_df))

# Reindex each DataFrame to ensure they all have the same number of rows
ecg_df = ecg_df.reindex(range(max_rows), fill_value=np.nan)
eye_tracking_df = eye_tracking_df.reindex(range(max_rows), fill_value=np.nan)
gsr_df = gsr_df.reindex(range(max_rows), fill_value=np.nan)

# Concatenate the features from all DataFrames (dropping the 'Label' column for now)
features_concat = pd.concat([
    ecg_df.drop(columns=['Label']), 
    eye_tracking_df.drop(columns=['Label']), 
    gsr_df.drop(columns=['Label'])
], axis=1)

# Re-add the 'Label' column
labels = ecg_df['Label'].fillna(method='ffill')  # Forward-fill label if any NaN

# Step for Iterative Imputation to handle NaNs
imputer = IterativeImputer(max_iter=10, random_state=0)  # You can adjust parameters as needed
features_imputed = pd.DataFrame(imputer.fit_transform(features_concat), columns=features_concat.columns)

# Check if there are still NaNs after imputation
print("Number of NaNs after imputation:", features_imputed.isna().sum().sum())  # Should print 0 if imputation worked

# Standardize the imputed features
scaler = StandardScaler()
features_standardized = pd.DataFrame(scaler.fit_transform(features_imputed), columns=features_imputed.columns)

# Number of runs and seed for reproducibility
num_runs = 3
random_seed = 42

# Store predictions and true labels across runs
all_true_labels = []
all_predictions = []
cumulative_confusion_matrix = np.zeros((len(labels.unique()), len(labels.unique())))
all_accuracies = []

# Run XGBoost multiple times with shuffling
for run in range(num_runs):
    # Shuffle the data
    X_train, X_test, y_train, y_test = train_test_split(
        features_standardized, labels, test_size=0.2, random_state=random_seed + run, shuffle=True
    )
    
    # Initialize XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Accumulate true labels and predictions
    all_true_labels.extend(y_test)
    all_predictions.extend(y_pred)
    
    # Accumulate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cumulative_confusion_matrix += cm
    
    # Calculate accuracy for this run
    accuracy = accuracy_score(y_test, y_pred)
    all_accuracies.append(accuracy)

# Final classification report based on all accumulated predictions
print("\nFinal Aggregated Classification Report Across 3 Runs:")
final_report = classification_report(all_true_labels, all_predictions)
print(final_report)

# Print the cumulative confusion matrix
print("\nCumulative Confusion Matrix Across 3 Runs:")
print(cumulative_confusion_matrix.astype(int))

# Print accuracies for each run in a single line
print("Accuracies for each run: ", " | ".join([f"Run {i+1}: {acc:.4f}" for i, acc in enumerate(all_accuracies)]))

# Print the final averaged accuracy across all runs
average_accuracy = np.mean(all_accuracies)
print(f"\nAveraged Accuracy Across 3 Runs: {average_accuracy:.4f}")

# Combine the standardized features with the Label column
final_data_with_labels = pd.concat([features_standardized, labels.reset_index(drop=True)], axis=1)
# Save the final imputed and standardized dataset with the label column to a CSV file
final_data_with_labels.to_csv('Concatenated_imputed_iterative.csv', index=False)
