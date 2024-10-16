%reset -f
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from keras.models import Model
from keras.layers import Dense, Input, Multiply

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

# Reindex the labels to match the features
labels_eye = labels_eye.reindex(range(max_rows), fill_value=labels_eye.mode()[0])
labels_gsr = labels_gsr.reindex(range(max_rows), fill_value=labels_gsr.mode()[0])
labels_ecg = labels_ecg.reindex(range(max_rows), fill_value=labels_ecg.mode()[0])

# Impute missing values
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

# Perform train-test split for each modality
X_train_eye, X_test_eye, y_train, y_test = train_test_split(features_eye_standardized, labels_eye, test_size=0.2, random_state=42)
X_train_gsr, X_test_gsr, _, _ = train_test_split(features_gsr_standardized, labels_gsr, test_size=0.2, random_state=42)
X_train_ecg, X_test_ecg, _, _ = train_test_split(features_ecg_standardized, labels_ecg, test_size=0.2, random_state=42)

# Attention mechanism for each modality
def attention_layer(inputs):
    attention_probs = Dense(inputs.shape[1], activation='softmax')(inputs)
    attention_mul = Multiply()([inputs, attention_probs])
    return attention_mul

# Apply attention on each modality (without LSTM, just dense layer)
input_eye = Input(shape=(X_train_eye.shape[1],))
input_gsr = Input(shape=(X_train_gsr.shape[1],))
input_ecg = Input(shape=(X_train_ecg.shape[1],))

attention_eye = attention_layer(input_eye)
attention_gsr = attention_layer(input_gsr)
attention_ecg = attention_layer(input_ecg)

# Build and compile models to extract attention-weighted features
model_eye = Model(inputs=input_eye, outputs=attention_eye)
model_gsr = Model(inputs=input_gsr, outputs=attention_gsr)
model_ecg = Model(inputs=input_ecg, outputs=attention_ecg)

# Extract attention-weighted features
attention_features_eye_train = model_eye.predict(X_train_eye)
attention_features_eye_test = model_eye.predict(X_test_eye)
attention_features_gsr_train = model_gsr.predict(X_train_gsr)
attention_features_gsr_test = model_gsr.predict(X_test_gsr)
attention_features_ecg_train = model_ecg.predict(X_train_ecg)
attention_features_ecg_test = model_ecg.predict(X_test_ecg)

# Train separate XGBoost models on attention-weighted features
print("Training XGBoost models on attention-weighted features...")
xgb_eye = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_gsr = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_ecg = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

xgb_eye.fit(attention_features_eye_train, y_train)
xgb_gsr.fit(attention_features_gsr_train, y_train)
xgb_ecg.fit(attention_features_ecg_train, y_train)

# Predict using the trained XGBoost models
y_pred_eye = xgb_eye.predict_proba(attention_features_eye_test)
y_pred_gsr = xgb_gsr.predict_proba(attention_features_gsr_test)
y_pred_ecg = xgb_ecg.predict_proba(attention_features_ecg_test)

# Soft Voting: Average the predictions from all modalities
y_pred_combined = (y_pred_eye + y_pred_gsr + y_pred_ecg) / 3
y_pred_classes = np.argmax(y_pred_combined, axis=1)

# Evaluate the combined predictions
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Confusion matrix, MCC, F1 Score, Balanced Accuracy
conf_matrix = confusion_matrix(y_test, y_pred_classes)
mcc = matthews_corrcoef(y_test, y_pred_classes)
f1_macro = f1_score(y_test, y_pred_classes, average='macro')
f1_weighted = f1_score(y_test, y_pred_classes, average='weighted')
balanced_acc = balanced_accuracy_score(y_test, y_pred_classes)

print("\nConfusion Matrix:")
print(conf_matrix)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"\nF1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")

# Save the final fused attention-weighted features as a single CSV
fused_train_data = pd.concat([pd.DataFrame(attention_features_eye_train), pd.DataFrame(attention_features_gsr_train), pd.DataFrame(attention_features_ecg_train)], axis=1)
fused_train_data['Label'] = y_train  # Use the labels from the training data

fused_test_data = pd.concat([pd.DataFrame(attention_features_eye_test), pd.DataFrame(attention_features_gsr_test), pd.DataFrame(attention_features_ecg_test)], axis=1)
fused_test_data['Label'] = y_test  # Use the labels from the testing data

# Combine the train and test data into one
fused_data = pd.concat([fused_train_data, fused_test_data])

# Save the fused data to a CSV file
fused_data.to_csv('attention_xgboost_hybrid_late.csv', index=False)
print("Fused data saved to 'attention_xgboost_hybrid_late.csv'.")
