import torch
from model import MNISTNeuralNet
import mlflow
from hyperparams import params
import os


trackinguri = "http://35.200.174.226:5000/"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'maximal-shadow-446510-n3-3117c795436e.json'


mlflow.set_tracking_uri(trackinguri)
client = mlflow.MlflowClient(tracking_uri=trackinguri)


def getModel():
    mnist_model = MNISTNeuralNet(hidden_dim=params["hidden_dim"],dropout_prob=params["dropout_prob"])


    model_name = "MNISTDigitRecognition"
    stage = "production" 

    latest_versions = client.get_latest_versions(name=model_name, stages=[stage])
    
    
    if latest_versions:


        latest_version = latest_versions[0]
        model_version = latest_version.version
        run_id = latest_version.run_id

        artifact_uri = client.get_run(run_id).info.artifact_uri
        
        dict_path = artifact_uri + "/model_weights/mnist_model_state_dict.pth"

        model_path = mlflow.artifacts.download_artifacts(artifact_uri=dict_path,dst_path='.')

        state_dict = torch.load(model_path,weights_only=False)
        mnist_model.load_state_dict(state_dict)
        model = mnist_model.eval()
        
        return model
    
    else : return None
