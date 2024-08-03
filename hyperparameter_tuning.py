import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data
data = load_iris()
X, y = data.data, data.target

# Define model
model = RandomForestClassifier()

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Define GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')

# Start MLflow run
with mlflow.start_run() as run:
    # Fit the model
    grid_search.fit(X, y)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("best_score", best_score)

    # Log model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
