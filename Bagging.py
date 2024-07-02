from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
import random
import numpy as np

def initialize_models():
    print("Inicializando Modelo")
    baggingLogisticRegression = BaggingClassifier(estimator=LogisticRegression(max_iter=2000), n_estimators=50, random_state=42, verbose=5)
    #baggingPerceptron = BaggingClassifier(estimator=Perceptron(), n_estimators=100, random_state=42)
    #baggingLDA = BaggingClassifier(estimator=LinearDiscriminantAnalysis(), n_estimators=100, random_state=42)
    return baggingLogisticRegression

def load_data(data):
    """Load and preprocess data."""
    print("Carregando dados")
    X_data, y_data = load_svmlight_file(data)
    X_data = X_data.toarray()  # Convert to dense array
    return X_data, y_data

def print_metrics(y_test, model_pred, model_proba):
    print("Imprimindo métricas")
    """Calculate and return the metrics for each model."""
    f1 = f1_score(y_test, model_pred, average='weighted')
    precision = precision_score(y_test, model_pred, average='weighted', zero_division=1)
    accuracy = accuracy_score(y_test, model_pred)
    auc = roc_auc_score(y_test, model_proba[:, 1])
    
    return f1, precision, accuracy, auc

def save_metrics_to_file(metrics, filename):
    print("Salvando métricas")
    """Save the metrics to a text file."""
    means = np.mean(metrics, axis=0)
    stds = np.std(metrics, axis=0)

    with open(filename, 'w') as f:
        f.write("Metric, Mean, Std\n")
        f.write(f"F1 Score, {means[0]}, {stds[0]}\n")
        f.write(f"Precision, {means[1]}, {stds[1]}\n")
        f.write(f"Accuracy, {means[2]}, {stds[2]}\n")
        f.write(f"AUC, {means[3]}, {stds[3]}\n")

def evaluate_model(model, X_data, y_data):
    print("Avaliando Modelo")
    # Define Repeated Stratified K-Fold cross-validator
    rsfk = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random.randint(0, 368512346))

    # Initialize list to store metrics
    all_metrics = []

    # Iterate over the folds
    for i, (train, test) in enumerate(rsfk.split(X_data, y_data)):
        # Split data
        X_train, y_train = X_data[train], y_data[train]
        X_test, y_test = X_data[test], y_data[test]

        # Fit models
        print("Fitting...")
        model.fit(X_train, y_train)

        # Predict
        print("Predict...")
        model_pred = model.predict(X_test)
        model_proba = model.predict_proba(X_test)

        # Calculate metrics
        metrics = print_metrics(y_test, model_pred, model_proba)
        all_metrics.append(metrics)

        # Print individual fold metrics (optional)
        print(f"Fold {i+1} metrics:")
        print(f"F1 Score: {metrics[0]}")
        print(f"Precision: {metrics[1]}")
        print(f"Accuracy: {metrics[2]}")
        print(f"AUC: {metrics[3]}")

    # Convert to numpy array for easier calculation of mean and std
    all_metrics = np.array(all_metrics)

    return all_metrics

def main(data, output_file):
    # Initialize models
    print("Iniciando modelo")
    baggingLogisticRegression = initialize_models()

    # Load data
    X_data, y_data = load_data(data)

    # Evaluate models
    lr_metrics = evaluate_model(baggingLogisticRegression, X_data, y_data)
    #perc_metrics = evaluate_model(baggingPerceptron, X_data, y_data)
    #lda_metrics = evaluate_model(baggingLDA, X_data, y_data)

    # Save metrics to file
    save_metrics_to_file(lr_metrics, "logistic_regression_" + output_file)
    #save_metrics_to_file(perc_metrics, "perceptron_" + output_file)
    #save_metrics_to_file(lda_metrics, "lda_" + output_file)

# Execute main function if the script is run directly
if __name__ == "__main__":
    main("cancer/libsvm/data_VGG.txt", "BLRVGG.txt")
