%reset -f
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input, concatenate
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

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

# Convert labels to categorical (one-hot encoding)
labels = to_categorical(labels_eye)

# Perform train-test split for each modality
X_train_eye, X_test_eye, y_train, y_test = train_test_split(features_eye_standardized, labels, test_size=0.2, random_state=42)
X_train_gsr, X_test_gsr, _, _ = train_test_split(features_gsr_standardized, labels, test_size=0.2, random_state=42)
X_train_ecg, X_test_ecg, _, _ = train_test_split(features_ecg_standardized, labels, test_size=0.2, random_state=42)

# Use np.expand_dims to add the extra dimension needed for CNN
X_train_eye = np.expand_dims(X_train_eye, axis=2)
X_test_eye = np.expand_dims(X_test_eye, axis=2)
X_train_gsr = np.expand_dims(X_train_gsr, axis=2)
X_test_gsr = np.expand_dims(X_test_gsr, axis=2)
X_train_ecg = np.expand_dims(X_train_ecg, axis=2)
X_test_ecg = np.expand_dims(X_test_ecg, axis=2)

# Build CNN models for each modality
input_eye = Input(shape=(X_train_eye.shape[1], 1))
cnn_eye = Conv1D(filters=64, kernel_size=3, activation='relu')(input_eye)
cnn_eye = MaxPooling1D(pool_size=2)(cnn_eye)
cnn_eye = Flatten()(cnn_eye)

input_gsr = Input(shape=(X_train_gsr.shape[1], 1))
cnn_gsr = Conv1D(filters=64, kernel_size=3, activation='relu')(input_gsr)
cnn_gsr = MaxPooling1D(pool_size=2)(cnn_gsr)
cnn_gsr = Flatten()(cnn_gsr)

input_ecg = Input(shape=(X_train_ecg.shape[1], 1))
cnn_ecg = Conv1D(filters=64, kernel_size=3, activation='relu')(input_ecg)
cnn_ecg = MaxPooling1D(pool_size=2)(cnn_ecg)
cnn_ecg = Flatten()(cnn_ecg)

# Concatenate the outputs from each CNN
merged = concatenate([cnn_eye, cnn_gsr, cnn_ecg])

# Fully connected layers after concatenation
dense = Dense(128, activation='relu')(merged)
output = Dense(y_train.shape[1], activation='softmax')(dense)

# Create the model
model = Model(inputs=[input_eye, input_gsr, input_ecg], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the CNN model...")
model.fit([X_train_eye, X_train_gsr, X_train_ecg], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate([X_test_eye, X_test_gsr, X_test_ecg], y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, balanced_accuracy_score

# Predictions and classification report
y_pred = model.predict([X_test_eye, X_test_gsr, X_test_ecg])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# Calculate and print the confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print(conf_matrix)
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")
# Calculate and print MCC
mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")

# Calculate and print F1 score (macro and weighted)
f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')
f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted')
print(f"\nF1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

# Calculate and print balanced accuracy
balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")

# Save the final fused data (train + test) as a single CSV
fused_train_data = pd.DataFrame(np.concatenate([X_train_eye.squeeze(), X_train_gsr.squeeze(), X_train_ecg.squeeze()], axis=1))
fused_train_data['Label'] = np.argmax(y_train, axis=1)  # Add the label column

fused_test_data = pd.DataFrame(np.concatenate([X_test_eye.squeeze(), X_test_gsr.squeeze(), X_test_ecg.squeeze()], axis=1))
fused_test_data['Label'] = np.argmax(y_test, axis=1)  # Add the label column

# Save the file using the fusion method and time
fused_data = pd.concat([fused_train_data, fused_test_data])
fused_data.to_csv('cnn_late.csv', index=False)
print("Fused data saved to 'cnn_late.csv'.")

