%reset -f
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight  # To calculate class weights

# Load the CSV files into pandas DataFrames
print("Loading data...")
eye_tracking_df = pd.read_csv('EyeTracking.csv')
gsr_df = pd.read_csv('GSR.csv')
ecg_df = pd.read_csv('ECG.csv')

# the first column is labeled consistently as 'Quad_Cat' for labels
labels_eye = eye_tracking_df['Quad_Cat'].fillna(method='ffill')

# Drop the label column from the feature set
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

# Convert the data to numpy arrays
X_eye = features_eye_standardized.values
X_gsr = features_gsr_standardized.values
X_ecg = features_ecg_standardized.values
y = labels_eye.values

# Reshape data for CNN (batch_size, channels, num_features)
X_eye = X_eye.reshape(X_eye.shape[0], 1, X_eye.shape[1])  # 1 channel for each modality
X_gsr = X_gsr.reshape(X_gsr.shape[0], 1, X_gsr.shape[1])
X_ecg = X_ecg.reshape(X_ecg.shape[0], 1, X_ecg.shape[1])

# Train-test split
X_eye_train, X_eye_test, X_gsr_train, X_gsr_test, X_ecg_train, X_ecg_test, y_train, y_test = train_test_split(
    X_eye, X_gsr, X_ecg, y, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_eye_train = torch.tensor(X_eye_train, dtype=torch.float32)
X_eye_test = torch.tensor(X_eye_test, dtype=torch.float32)
X_gsr_train = torch.tensor(X_gsr_train, dtype=torch.float32)
X_gsr_test = torch.tensor(X_gsr_test, dtype=torch.float32)
X_ecg_train = torch.tensor(X_ecg_train, dtype=torch.float32)
X_ecg_test = torch.tensor(X_ecg_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Calculate class weights to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Define the CNN + LSTM model
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        
        # CNN for each modality (1D convolution)
        self.cnn_eye = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 1 channel, 32 filters
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.cnn_gsr = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.cnn_ecg = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM for each modality
        self.lstm_eye = nn.LSTM(32, 64, batch_first=True)  # 32 input size from CNN
        self.lstm_gsr = nn.LSTM(32, 64, batch_first=True)
        self.lstm_ecg = nn.LSTM(32, 64, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(64 * 3, 4)  # 3 modalities, 4 classes
        
    def forward(self, X_eye, X_gsr, X_ecg):
        # CNN for each modality
        X_eye = self.cnn_eye(X_eye)
        X_gsr = self.cnn_gsr(X_gsr)
        X_ecg = self.cnn_ecg(X_ecg)
        
        # LSTM for each modality
        _, (X_eye, _) = self.lstm_eye(X_eye)
        _, (X_gsr, _) = self.lstm_gsr(X_gsr)
        _, (X_ecg, _) = self.lstm_ecg(X_ecg)
        
        # Concatenate LSTM outputs from all modalities
        X = torch.cat((X_eye[-1], X_gsr[-1], X_ecg[-1]), dim=1)
        
        # Fully connected layer for final classification
        out = self.fc(X)
        return out

# Initialize the model, loss function, and optimizer
model = CNNLSTM()
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # Applying class weights here
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_eye_train, X_gsr_train, X_ecg_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing the model
model.eval()
with torch.no_grad():
    y_pred = model(X_eye_test, X_gsr_test, X_ecg_test)
    _, predicted = torch.max(y_pred, 1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predicted))

    # Accuracy and confusion matrix
    accuracy = accuracy_score(y_test, predicted)
    conf_matrix = confusion_matrix(y_test, predicted)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Additional metrics: F1 score, MCC, and balanced accuracy
    mcc = matthews_corrcoef(y_test, predicted)
    f1_macro = f1_score(y_test, predicted, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, predicted)

    print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")



# Save the final concatenated features (fused) into a CSV file
final_fused_data = pd.DataFrame(np.concatenate([X_eye.reshape(X_eye.shape[0], -1), 
                                                X_gsr.reshape(X_gsr.shape[0], -1), 
                                                X_ecg.reshape(X_ecg.shape[0], -1)], axis=1))

final_fused_data['Label'] = labels_eye.values  # Re-add the labels to the dataset

# Save to CSV with a meaningful name
final_fused_data.to_csv('cnn_lstm_hybrid_fusion.csv', index=False)
print("Fused data saved to 'cnn_lstm_hybrid_fusion.csv'")
