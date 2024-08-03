import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_hyperparameter_tuning():
    with mlflow.start_run():
        # Load the dataset
        df = pd.read_csv('Iris.csv')

        # Split the dataset into features and target variable
        X = df.drop(columns=['Species'])
        y = df['Species']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        clf = RandomForestClassifier()

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Log parameters
        mlflow.log_params(best_params)

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Print results
        print(f"Best Parameters: {best_params}")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

        # Print MLflow run information
        print(f"Run ID: {mlflow.active_run().info.run_id}")

# Run the hyperparameter tuning process
if __name__ == "__main__":
    run_hyperparameter_tuning()
