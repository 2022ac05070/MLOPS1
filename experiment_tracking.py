import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Set the MLflow tracking URI to a local directory
mlflow.set_tracking_uri("file:///tmp/mlruns")


# Define a function to run an experiment
def run_experiment(n_estimators, max_depth):

    with mlflow.start_run():

        # Load the dataset
        df = pd.read_csv('Iris.csv')

        # Split the dataset into features and target variable
        X = df.drop(columns=['Species'])
        y = df['Species']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Save the model
        dump(clf, 'model.joblib')

        # Add hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        clf = grid_search.best_estimator_

        # Log the best parameters from grid search
        best_params = grid_search.best_params_
        mlflow.log_param("best_n_estimators", best_params['n_estimators'])
        mlflow.log_param("best_max_depth", best_params['max_depth'])

        # Calculate accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(clf, "model")

        # Print results
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

        # Print MLflow run information
        print(f"Run ID: {mlflow.active_run().info.run_id}")


# Run experiments with different hyperparameters
run_experiment(n_estimators=100, max_depth=None)
run_experiment(n_estimators=150, max_depth=10)
run_experiment(n_estimators=200, max_depth=20)
