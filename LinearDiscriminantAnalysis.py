from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import random

def initialize_models():
    """
    Initialize the LinearDiscriminantAnalysis model.
    
    Returns:
        LinearDiscriminantAnalysis: Initialized LDA model.
    """
    lda = LinearDiscriminantAnalysis()
    return lda

def load_data(data_path):
    """
    Load and preprocess data from a specified file path.
    
    Args:
        data_path (str): Path to the data file.
    
    Returns:
        tuple: Features (X_data) and labels (y_data).
    """
    X_data, y_data = load_svmlight_file(data_path)
    X_data = X_data.toarray()  # Convert to dense array
    return X_data, y_data

def compute_metrics(y_test, model_pred, model_proba):
    """
    Compute accuracy, F1 score, precision, and AUC for the given predictions.
    
    Args:
        y_test (array): True labels.
        model_pred (array): Predicted labels.
        model_proba (array): Predicted probabilities.
    
    Returns:
        tuple: Accuracy, F1 score, precision, and AUC.
    """
    accuracy = accuracy_score(y_test, model_pred)
    f1 = f1_score(y_test, model_pred, average='weighted')
    precision = precision_score(y_test, model_pred, average='weighted', zero_division=1)
    auc = roc_auc_score(y_test, model_proba, multi_class='ovo', average='weighted')
    return accuracy, f1, precision, auc

def print_metrics(y_test, model_pred, model_proba, model_name):
    """
    Print and return the metrics for a given model's predictions.
    
    Args:
        y_test (array): True labels.
        model_pred (array): Predicted labels.
        model_proba (array): Predicted probabilities.
        model_name (str): Name of the model.
    
    Returns:
        tuple: Accuracy, F1 score, precision, and AUC.
    """
    accuracy, f1, precision, auc = compute_metrics(y_test, model_pred, model_proba)
    print(f'Accuracy {model_name}: {accuracy}')
    print(f'F1 Score {model_name}: {f1}')
    print(f'Precision {model_name}: {precision}')
    print(f'AUC {model_name}: {auc}')
    return accuracy, f1, precision, auc

def main(data_path):
    # Initialize models
    lda = initialize_models()

    # Load data
    X_data, y_data = load_data(data_path)

    # Define Repeated Stratified K-Fold cross-validator with fixed random state for reproducibility
    random_state = random.randint(0, 368512346)
    rsfk = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)

    # Lists to store metrics
    accuracies = []
    f1_scores = []
    precisions = []
    aucs = []

    # Iterate over the folds
    for train_index, test_index in rsfk.split(X_data, y_data):
        # Split data
        X_train, y_train = X_data[train_index], y_data[train_index]
        X_test, y_test = X_data[test_index], y_data[test_index]

        # Fit model
        lda.fit(X_train, y_train)

        # Predict
        lda_pred = lda.predict(X_test)
        lda_proba = lda.predict_proba(X_test)

        # Print and store metrics
        accuracy, f1, precision, auc = print_metrics(y_test, lda_pred, lda_proba, 'LDA')
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        aucs.append(auc)

    # Calculate averages and standard deviations
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_f1_score = np.mean(f1_scores)
    std_f1_score = np.std(f1_scores)
    avg_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Save metrics to a text file
    with open("resultsLDAXception.txt", "w") as file:
        file.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        file.write(f"Standard Deviation of Accuracy: {std_accuracy:.4f}\n")
        file.write(f"Average F1 Score: {avg_f1_score:.4f}\n")
        file.write(f"Standard Deviation of F1 Score: {std_f1_score:.4f}\n")
        file.write(f"Average Precision: {avg_precision:.4f}\n")
        file.write(f"Standard Deviation of Precision: {std_precision:.4f}\n")
        file.write(f"Average AUC: {avg_auc:.4f}\n")
        file.write(f"Standard Deviation of AUC: {std_auc:.4f}\n")

# Execute main function if the script is run directly
if __name__ == "__main__":
    main("cancer/libsvm/data_Xception.txt")
