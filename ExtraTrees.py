from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import label_binarize
import random
import numpy as np

def initialize_models():
    # Initialize Extra Trees with some manually set hyperparameters
    return ExtraTreesClassifier()

def load_data(data_path):
    """Load and preprocess data."""
    X_data, y_data = load_svmlight_file(data_path)
    X_data = X_data.toarray()  # Convert to dense array
    return X_data, y_data

def print_metrics(y_true, y_pred, y_score, model_name, classes):
    """Print the metrics for each model and return them as a dictionary."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    
    # Calculate AUC
    if len(classes) == 2:
        auc = roc_auc_score(y_true, y_score[:, 1])
    else:
        y_true_binarized = label_binarize(y_true, classes=classes)
        auc = roc_auc_score(y_true_binarized, y_score, average="weighted", multi_class="ovr")
    
    print(f'\nMetrics for {model_name}:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'AUC: {auc:.4f}')
    print('Classification Report:')
    print(classification_report(y_true, y_pred, zero_division=1))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc}

def save_metrics_to_file(metrics, filename):
    """Save the mean and standard deviation of metrics to a file."""
    with open(filename, 'w') as f:
        for metric_name, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            f.write(f'{metric_name}: Mean = {mean_value:.4f}, Std = {std_value:.4f}\n')


def main(data_path, output_file):
    # Initialize models
    et = initialize_models()

    # Load data
    X_data, y_data = load_data(data_path)
    classes = np.unique(y_data)  # Get unique classes

    # Define Repeated Stratified K-Fold cross-validator
    rsfk = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random.randint(0, 368512346))

    # Initialize dictionary to store metrics
    metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}

    # Iterate over the folds
    for fold_idx, (train_idx, test_idx) in enumerate(rsfk.split(X_data, y_data)):
        print(f'\nFold {fold_idx + 1}:')

        # Split data
        X_train, y_train = X_data[train_idx], y_data[train_idx]
        X_test, y_test = X_data[test_idx], y_data[test_idx]

        # Fit the model
        et.fit(X_train, y_train)

        # Predict
        et_pred = et.predict(X_test)
        et_proba = et.predict_proba(X_test)

        # Print and collect metrics
        fold_metrics = print_metrics(y_test, et_pred, et_proba, 'et', classes)
        for metric_name, value in fold_metrics.items():
            metrics[metric_name].append(value)
    
    # Save the mean and standard deviation of metrics to a file
    save_metrics_to_file(metrics, output_file)

# Execute main function if the script is run directly
if __name__ == "__main__":
    main("cancer/libsvm/data_Xception.txt", "resultsETXception.txt")
