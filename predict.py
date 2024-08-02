import pandas as pd
from joblib import load

# Load the dataset
df = pd.read_csv('iris.csv')

# Load the trained model
clf = load('model.joblib')

# Make predictions
X = df.drop(columns=['Species'])
predictions = clf.predict(X)

# Output predictions
print(predictions)
