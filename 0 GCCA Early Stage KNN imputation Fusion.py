%reset -f
import pandas as pd
import numpy as np
from mvlearn.embed import GCCA  # Generalized Canonical Correlation Analysis from mvlearn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load the CSV files into pandas DataFrames
ecg_df = pd.read_csv('ECG.csv')
eye_tracking_df = pd.read_csv('EyeTracking.csv')
gsr_df = pd.read_csv('GSR.csv')

# the first column is labeled consistently as 'Label'
ecg_df.columns = ['Label'] + list(ecg_df.columns[1:])
eye_tracking_df.columns = ['Label'] + list(eye_tracking_df.columns[1:])
gsr_df.columns = ['Label'] + list(gsr_df.columns[1:])

# Padding DataFrames to the same number of rows (using NaN where data is missing)
max_rows = max(len(ecg_df), len(eye_tracking_df), len(gsr_df))

# Reindex each DataFrame to all have the same number of rows
ecg_df = ecg_df.reindex(range(max_rows), fill_value=np.nan)
eye_tracking_df = eye_tracking_df.reindex(range(max_rows), fill_value=np.nan)
gsr_df = gsr_df.reindex(range(max_rows), fill_value=np.nan)

# Separate features (dropping the 'Label' column)
ecg_features = ecg_df.drop(columns=['Label'])
eye_tracking_features = eye_tracking_df.drop(columns=['Label'])
gsr_features = gsr_df.drop(columns=['Label'])

# Handle missing labels by forward-filling
labels = ecg_df['Label'].fillna(method='ffill')

# Imputation to handle NaNs in features
imputer = KNNImputer(n_neighbors=5)
ecg_features_imputed = pd.DataFrame(imputer.fit_transform(ecg_features), columns=ecg_features.columns)
eye_tracking_features_imputed = pd.DataFrame(imputer.fit_transform(eye_tracking_features), columns=eye_tracking_features.columns)
gsr_features_imputed = pd.DataFrame(imputer.fit_transform(gsr_features), columns=gsr_features.columns)

# Standardize the features
scaler = StandardScaler()
ecg_standardized = scaler.fit_transform(ecg_features_imputed)
eye_standardized = scaler.fit_transform(eye_tracking_features_imputed)
gsr_standardized = scaler.fit_transform(gsr_features_imputed)

# Apply Generalized CCA (GCCA) using the mvlearn library
gcca = GCCA(n_components=5)  # You can adjust the number of components

# Fit GCCA on the three views (ECG, EyeTracking, GSR)
views = [ecg_standardized, eye_standardized, gsr_standardized]
gcca_transformed_views = gcca.fit_transform(views)

# Combine all GCCA-transformed views
features_fused = np.concatenate(gcca_transformed_views, axis=1)

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
        features_fused, labels, test_size=0.2, random_state=random_seed + run, shuffle=True
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

# Save the GCCA-transformed data with the labels
final_data_with_labels = pd.concat([pd.DataFrame(features_fused), labels.reset_index(drop=True)], axis=1)
final_data_with_labels.to_csv('GCCA_fused_with_labels.csv', index=False)
