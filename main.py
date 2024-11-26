import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import joblib
import os 


# create application Flask
app = Flask(__name__)

#verify if the model already was trained and saved
MODEL_PATH = 'diabetes_model.pkl'

# Function to train the model
def train_model():
  #load dataset 
  url = './pima-indians-diabetes.data.csv'
  columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
               'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
  
  data = pd.read_csv(url, header=None, names=columns)
  
  # spliting independent and dependent variables
  X = data.drop('Outcome', axis=1)
  y = data['Outcome']
  
  # dividing training and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # scaling of data
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # creating and training the model
  model = LogisticRegression()
  model.fit(X_train_scaled, y_train)
  
  # evaluate the model
  accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
  print(f'Acuracy the model: {accuracy:.2f}')
  
  #saving the trained model 
  joblib.dump(model, MODEL_PATH)
  joblib.dump(scaler, 'scaler.pkl') # saving scaler to user in predictions
  
  return model


# function to load the trained model 
def load_model():
  if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load('scaler.pkl')
    return model, scaler
  else:
    # to train the model if not exist yet
    return train_model(), StandardScaler()
  
#route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
  # load the trained model
  model, scaler = load_model()
  
  # receive data from request
  data = request.get_json() # expect data in json format
  input_data = np.array([data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                           data['SkinThickness'], data['Insulin'], data['BMI'],
                           data['DiabetesPedigreeFunction'], data['Age']]).reshape(1, -1)
  
  # scale input data with the same scaler used in the train
  input_data_scaled = scaler.transform(input_data)
  
  # making prediction
  prediction = model.predict(input_data_scaled)
  
  #return the result 
  return jsonify({'prediction': int(prediction[0])}) # return 1 if diabetes or 0 if not diabetes
  

# route to train the model again
@app.route('/train', methods=["GET"])
def retrain():
  train_model()
  return jsonify({'message': 'Model trained and saved successfully.'})

if __name__ == '__main__':
  app.run(debug=True)