%reset -f
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier

# Load CSVs into DataFrames
print("Loading data...")
eye_tracking_df = pd.read_csv('EyeTracking.csv')
gsr_df = pd.read_csv('GSR.csv')
ecg_df = pd.read_csv('ECG.csv')

# Labels
labels_eye = eye_tracking_df['Quad_Cat'].fillna(method='ffill')

# Drop the label column
features_eye = eye_tracking_df.drop(columns=['Quad_Cat'])
features_gsr = gsr_df.drop(columns=['Quad_Cat'])
features_ecg = ecg_df.drop(columns=['Quad_Cat'])

# Align the number of rows across modalities by padding with NaN
print("Aligning the number of rows across modalities...")
max_rows = max(len(features_eye), len(features_gsr), len(features_ecg))
features_eye = features_eye.reindex(range(max_rows), fill_value=np.nan)
features_gsr = features_gsr.reindex(range(max_rows), fill_value=np.nan)
features_ecg = features_ecg.reindex(range(max_rows), fill_value=np.nan)
labels_eye = labels_eye.reindex(range(max_rows), fill_value=labels_eye.mode()[0])

# Impute missing values for each modality
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
features_eye_imputed = pd.DataFrame(imputer.fit_transform(features_eye), columns=features_eye.columns)
features_gsr_imputed = pd.DataFrame(imputer.fit_transform(features_gsr), columns=features_gsr.columns)
features_ecg_imputed = pd.DataFrame(imputer.fit_transform(features_ecg), columns=features_ecg.columns)

# Standardize the features for each modality
print("Standardizing features...")
scaler = StandardScaler()
features_eye_standardized = pd.DataFrame(scaler.fit_transform(features_eye_imputed), columns=features_eye_imputed.columns)
features_gsr_standardized = pd.DataFrame(scaler.fit_transform(features_gsr_imputed), columns=features_gsr_imputed.columns)
features_ecg_standardized = pd.DataFrame(scaler.fit_transform(features_ecg_imputed), columns=features_ecg_imputed.columns)

# Apply PCA separately to each modality
print("Applying PCA to each modality...")
pca_eye = PCA(n_components=5)  # Adjust n_components based on data
pca_gsr = PCA(n_components=5)
pca_ecg = PCA(n_components=5)

reduced_eye = pca_eye.fit_transform(features_eye_standardized)
reduced_gsr = pca_gsr.fit_transform(features_gsr_standardized)
reduced_ecg = pca_ecg.fit_transform(features_ecg_standardized)

# Combine the reduced features from all modalities
combined_reduced_features = np.concatenate([reduced_eye, reduced_gsr, reduced_ecg], axis=1)

# Perform LDA for class separation on the combined PCA-reduced features
print("Applying LDA...")
lda = LDA(n_components=2)  # Adjust this based on the number of classes
lda_features = lda.fit_transform(combined_reduced_features, labels_eye)

# Train-test split for LDA features
X_train, X_test, y_train, y_test = train_test_split(lda_features, labels_eye, test_size=0.2, random_state=42)

# Train a classifier (RandomForest in this case)
print("Training RandomForest classifier...")
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict using the trained classifier
y_pred = classifier.predict(X_test)

# Evaluate the predictions
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Confusion matrix, MCC, F1 Score, Balanced Accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"\nF1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")

# Save the final reduced PCA + LDA features into a CSV file
final_fused_data = pd.DataFrame(lda_features)
final_fused_data['Label'] = labels_eye.values  # Add the labels to the dataset

# Save to CSV with a meaningful name reflecting hybrid fusion
final_fused_data.to_csv('pca_lda_hybrid_fusion.csv', index=False)
print("Fused data saved to 'pca_lda_hybrid_fusion.csv'")
