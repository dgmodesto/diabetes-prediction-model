import requests

# Dados de entrada para a previsão (com diabetes)
data_diabetes = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}

# Dados de entrada para a previsão (sem diabetes)
data_sem_diabetes = {
    "Pregnancies": 2,
    "Glucose": 85,
    "BloodPressure": 66,
    "SkinThickness": 29,
    "Insulin": 0,
    "BMI": 26.6,
    "DiabetesPedigreeFunction": 0.351,
    "Age": 29
}

# Função para testar a API
def test_prediction(data):
    response = requests.post('http://127.0.0.1:5000/predict', json=data)
    return response.json()

# Testando com dados de pacientes com diabetes
print("Resultado para paciente com diabetes:")
print(test_prediction(data_diabetes))

# Testando com dados de pacientes sem diabetes
print("\nResultado para paciente sem diabetes:")
print(test_prediction(data_sem_diabetes))
