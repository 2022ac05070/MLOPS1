import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
df = pd.read_csv('Iris.csv')

# Split the dataset into features and target variable
X = df.drop(columns=['Species'])
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save the model
dump(clf, 'model.joblib')
