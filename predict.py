import joblib
import pandas as pd

def predict():
    # Load model
    model = joblib.load('models/iris_model.pkl')
    
    # Sample data for prediction
    data = pd.DataFrame({
        'SepalLengthCm': [5.1],
        'SepalWidthCm': [3.5],
        'PetalLengthCm': [1.4],
        'PetalWidthCm': [0.2]
    })
    
    # Make prediction
    prediction = model.predict(data)
    print(f'Prediction: {prediction}')

if __name__ == '__main__':
    predict()
