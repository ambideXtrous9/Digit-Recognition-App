import mlflow
import numpy as np
from mlflow import MlflowClient


trackinguri = "http://127.0.0.1:5000/"

mlflow.set_tracking_uri(trackinguri)
client = MlflowClient(tracking_uri=trackinguri)

# Load the model
model_name = "MNIST-Digit-Recognizer"
stage = "Production"

# Get the latest version of the model in the specified stage
latest_versions = client.get_latest_versions(name=model_name, stages=[stage])

# Print the model name and version
for version in latest_versions:
    print(f"\nModel Name: {version.name}, Version: {version.version}\n")

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")


def predict(image):
    pred = loaded_model.predict(image)
    # Get the max probability
    max_prob = np.max(pred)
    # Get the index of the max probability
    max_prob_index = np.argmax(pred)
    
    print(f"\nPrediction Accuracy = {max_prob} || Predicted Digit = {max_prob_index}\n")
    
    return max_prob,max_prob_index



