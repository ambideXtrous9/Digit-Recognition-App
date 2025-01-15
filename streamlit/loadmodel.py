import torch
from model import MNISTNeuralNet
import mlflow
from hyperparams import params


trackinguri = "http://127.0.0.1:5000/"

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

        model_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri,
                                                        dst_path='statedict')


        ml_path = 'statedict/artifacts/model_weights/mnist_model_state_dict.pth'

        state_dict = torch.load(ml_path)
        mnist_model.load_state_dict(state_dict)
        model = mnist_model.eval()
        
        return model
    
    else : return None
