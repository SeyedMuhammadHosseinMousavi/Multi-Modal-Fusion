%reset -f
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, f1_score, balanced_accuracy_score

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

# Reindex each DataFrame. all have the same number of rows
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

# Build autoencoder for each modality
def build_autoencoder(input_shape):
    input_layer = Input(shape=(input_shape,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    latent_space = Dense(16, activation='relu')(encoded)  # Latent space representation
    decoded = Dense(32, activation='relu')(latent_space)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_shape, activation='linear')(decoded)
    
    # Autoencoder model (for training)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    
    # Encoder model (for extracting latent features)
    encoder = Model(inputs=input_layer, outputs=latent_space)
    
    return autoencoder, encoder

# Build autoencoders for each modality
autoencoder_eye, encoder_eye = build_autoencoder(X_train_eye.shape[1])
autoencoder_gsr, encoder_gsr = build_autoencoder(X_train_gsr.shape[1])
autoencoder_ecg, encoder_ecg = build_autoencoder(X_train_ecg.shape[1])

# Compile autoencoders
autoencoder_eye.compile(optimizer='adam', loss='mse')
autoencoder_gsr.compile(optimizer='adam', loss='mse')
autoencoder_ecg.compile(optimizer='adam', loss='mse')

# Train autoencoders
print("Training autoencoders...")
autoencoder_eye.fit(X_train_eye, X_train_eye, epochs=50, batch_size=32, validation_split=0.2)
autoencoder_gsr.fit(X_train_gsr, X_train_gsr, epochs=50, batch_size=32, validation_split=0.2)
autoencoder_ecg.fit(X_train_ecg, X_train_ecg, epochs=50, batch_size=32, validation_split=0.2)

# Extract latent features from the encoders
print("Extracting latent features...")
latent_eye_train = encoder_eye.predict(X_train_eye)
latent_gsr_train = encoder_gsr.predict(X_train_gsr)
latent_ecg_train = encoder_ecg.predict(X_train_ecg)

latent_eye_test = encoder_eye.predict(X_test_eye)
latent_gsr_test = encoder_gsr.predict(X_test_gsr)
latent_ecg_test = encoder_ecg.predict(X_test_ecg)

# Concatenate the latent representations for all modalities
print("Concatenating latent features...")
latent_train_combined = np.concatenate([latent_eye_train, latent_gsr_train, latent_ecg_train], axis=1)
latent_test_combined = np.concatenate([latent_eye_test, latent_gsr_test, latent_ecg_test], axis=1)

# Build the classification model using the latent features
input_layer = Input(shape=(latent_train_combined.shape[1],))
dense_layer = Dense(128, activation='relu')(input_layer)
output_layer = Dense(y_train.shape[1], activation='softmax')(dense_layer)

classifier = Model(inputs=input_layer, outputs=output_layer)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classifier
print("Training classifier with concatenated latent features...")
classifier.fit(latent_train_combined, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the classifier
print("Evaluating the classifier...")
test_loss, test_accuracy = classifier.evaluate(latent_test_combined, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = classifier.predict(latent_test_combined)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# Confusion matrix, MCC, F1 Score, Balanced Accuracy
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')
f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted')
balanced_acc = balanced_accuracy_score(y_true_classes, y_pred_classes)

print("\nConfusion Matrix:")
print(conf_matrix)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"\nF1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")

# Save the final fused data (train + test) as a single CSV
fused_data = pd.DataFrame(np.concatenate([latent_train_combined, latent_test_combined], axis=0))
fused_data['Label'] = np.concatenate([np.argmax(y_train, axis=1), np.argmax(y_test, axis=1)])  # Add the label column

# Save the file using the fusion method and time
fused_data.to_csv('autoencoder_late.csv', index=False)
print("Fused data saved to 'autoencoder_late.csv'.")

