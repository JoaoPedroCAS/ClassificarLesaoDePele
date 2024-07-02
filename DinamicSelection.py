import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from deslib.des.meta_des import METADES
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.static.stacked import StackedClassifier
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import os 

rng = np.random.RandomState(42)

def initialize_models():
    print("iniciando modelo")
    # Initialize DecisionTree with some manually set hyperparameters
    return BaggingClassifier(estimator=Perceptron(), n_estimators=50, verbose=5), BaggingClassifier(estimator=Perceptron(), n_estimators=100, verbose=5)

def load_data(data_path):
    print("Carregando dados")
    """Load and preprocess data."""
    X_data, y_edata = load_svmlight_file(data_path)
    X_data = X_data.toarray()  # Convert to dense array
    return X_data, y_data

def print_metrics(y_true, y_pred, y_score, model_name, classes):
    print("Mostrando métricas")
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
    print("Salvando métricas")
    """Save the mean and standard deviation of metrics to a file."""
    with open(filename, 'w') as f:
        for metric_name, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            f.write(f'{metric_name}: Mean = {mean_value:.4f}, Std = {std_value:.4f}\n')

def main(data_path, output_file):
    rf, rf1 = initialize_models()

    X_data, y_data = load_data(data_path)
    classes = np.unique(y_data)

    rsfk = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=rng)

    metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}

    for fold_idx, (train_idx, test_idx) in enumerate(rsfk.split(X_data, y_data)):
        print(f'\nFold {fold_idx + 1}:')

        X_train, y_train = X_data[train_idx], y_data[train_idx]
        X_test, y_test = X_data[test_idx], y_data[test_idx]

        rf.fit(X_train, y_train)
        rf1.fit(X_train, y_train)
        knorau = KNORAU([rf, rf1])
        kne = KNORAE([rf, rf1])
        #meta = METADES([rf, rf1])
        
        knorau.fit(X_train, y_train)
        kne.fit(X_train, y_train)
        #meta.fit(X_train, y_train)    

        for model, model_name in zip([rf, rf1, knorau, kne], ['Perceptron10', 'Perceptron20', 'KNORAU', 'KNORAE']):
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)

            fold_metrics = print_metrics(y_test, y_pred, y_score, model_name, classes)
            for metric_name, value in fold_metrics.items():
                metrics[metric_name].append(value)

    save_metrics_to_file(metrics, output_file)

if __name__ == "__main__":
    main("cancer/libsvm/data_VGG.txt", "metrics_output.txt")
