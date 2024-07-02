import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import vgg19, inception_v3, xception
from pathlib import Path

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress informational messages

# Define paths
drive_path = Path('C:/Users/jpedr/OneDrive/Documentos/TCC/Codigos/CancerDePele/cancer/')
data_file_path = drive_path / 'data.txt'
data_dir = drive_path / 'data'

# Initialize lists for storing data
X = []
y = []

# Read data from file and process each line
with open(data_file_path, 'r') as file:
    for line in file:
        # Split line into name and class_label
        name, class_label = line.split()
        
        # Construct image path and read image
        img_path = data_dir / name
        img = cv2.imread(str(img_path))
        
        # Resize image and append to list
        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(class_label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Convert labels to categorical
num_classes = len(le.classes_)
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

def build_model(base_model, num_classes):
    # Freeze layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Combine base model with custom head
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=5):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
    return history

def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred_classes, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)

    # Calculate ROC AUC for each class and the macro average
    y_test_bin = preprocessing.label_binarize(y_true, classes=range(num_classes))
    roc_auc = roc_auc_score(y_test_bin, y_pred, average='macro', multi_class='ovr')

    return {
        "Accuracy": accuracy_score(y_true, y_pred_classes),
        "Precision": precision_score(y_true, y_pred_classes, average='weighted', zero_division=1),
        "Recall": recall_score(y_true, y_pred_classes, average='weighted', zero_division=1),
        "F1-Score": f1_score(y_true, y_pred_classes, average='weighted', zero_division=1),
        "ROC AUC": roc_auc,
    }

# Models
available_models = {
    "VGG19": vgg19.VGG19,
    "Inception": inception_v3.InceptionV3,
    "Xception": xception.Xception
}

# Select the model
selected_model_name = "Xception"
selected_model = available_models[selected_model_name](weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("Modelo Escolhido com sucesso!")

# Repeated Stratified K-Fold Cross-Validation
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

# Convert labels back to single dimension for splitting
y_single = np.argmax(y, axis=1)

# Metrics storage
accuracies = []
precisions = []
f1_scores = []
roc_aucs = []

fold = 0
for train_index, test_index in rskf.split(X, y_single):
    fold += 1
    print(f"Fold {fold}")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Build the model
    model = build_model(selected_model, num_classes)
    print("Modelo Construido com sucesso!")

    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test)
    print("Modelo Treinado com sucesso!")

    # Evaluate the model
    evaluation = evaluate_model(model, X_test, y_test, le)
    print("Modelo Avaliado com sucesso!")

    # Collect metrics
    accuracies.append(evaluation["Accuracy"])
    precisions.append(evaluation["Precision"])
    f1_scores.append(evaluation["F1-Score"])
    roc_aucs.append(evaluation["ROC AUC"])

    # Print overall metrics for current fold
    print(f"\nOverall Metrics for {selected_model_name} - Fold {fold}:")
    for metric, value in evaluation.items():
        print(metric + ":", value)

    # Plot training history
    #plt.plot(history.history['accuracy'], label='Acurácia')
    #plt.plot(history.history['val_accuracy'], label='Acurácia de validação')
    #plt.title(f'Acurácia - {selected_model_name} - Fold {fold}')
    #plt.ylabel('Acurácia')
    #plt.xlabel('Época')
    #plt.legend(loc='lower right')
    #plt.show()

# Calculate mean and standard deviation
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
mean_roc_auc = np.mean(roc_aucs)
std_roc_auc = np.std(roc_aucs)

# Save results to text file
results_path = drive_path / 'resultsCNNXception.txt'
with open(results_path, 'w') as f:
    f.write(f"Model: {selected_model_name}\n")
    f.write(f"Accuracy: Mean = {mean_accuracy:.4f}, Std = {std_accuracy:.4f}\n")
    f.write(f"Precision: Mean = {mean_precision:.4f}, Std = {std_precision:.4f}\n")
    f.write(f"F1-Score: Mean = {mean_f1:.4f}, Std = {std_f1:.4f}\n")
    f.write(f"ROC AUC: Mean = {mean_roc_auc:.4f}, Std = {std_roc_auc:.4f}\n")

print("Results saved successfully!")
