from tqdm import tqdm
import torch
from torch import nn
from model import MNISTNeuralNet
import mlflow
from torchinfo import summary
from DataLoader import train_val_dataloader
from hyperparams import params



trackinguri = "http://127.0.0.1:5000/"

mlflow.set_tracking_uri(trackinguri)
client = mlflow.MlflowClient(tracking_uri=trackinguri)

MLFLOW_EXPERIMENT_NAME = "Digit-Recognition"

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
exp_id = experiment.experiment_id if experiment else mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNISTNeuralNet(hidden_dim=params["hidden_dim"],dropout_prob=params["dropout_prob"])

model.to(device)

state_dict_path = "mnist_model_state_dict.pth"


loss_fnc = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.AdamW(lr=params["learning_rate"],params=model.parameters())

train_dataloader, val_dataloader = train_val_dataloader(batch_size=params["batch_size"])

def trainFunc(epoch,model,train_dataloader):
    model.train()
    epoch_loss = 0
    total_correct = 0

    for batch_idx,(data,target) in enumerate(tqdm(train_dataloader)):
        (data,target) = (data.to(device),target.to(device))

        optimizer.zero_grad()

        pred = model(data)
        loss = loss_fnc(pred,target)

        loss.backward()
        optimizer.step()

        epoch_loss = epoch_loss + (loss.item()*target.shape[0])
        _,output = torch.max(pred,dim=1)
        total_correct = total_correct + (output==target).sum().item()



    epoch_loss = epoch_loss / len(train_dataloader.dataset)
    total_correct = total_correct / len(train_dataloader.dataset)

    print(f"Epoch = {epoch} || Train Loss = {epoch_loss} || Train Accuracy = {total_correct}")
    return epoch_loss,total_correct


def valFunc(epoch,model,val_dataloader):
    model.eval()
    epoch_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(tqdm(val_dataloader)):
            (data,target) = (data.to(device),target.to(device))

            pred = model(data)
            loss = loss_fnc(pred,target)

            epoch_loss = epoch_loss + (loss.item()*target.shape[0])
            _,output = torch.max(pred,dim=1)
            total_correct = total_correct + (output==target).sum().item()

    epoch_loss = epoch_loss / len(val_dataloader.dataset)
    total_correct = total_correct / len(val_dataloader.dataset)

    print(f"Epoch = {epoch} || Val Loss = {epoch_loss} || Val Accuracy = {total_correct}")
    return epoch_loss,total_correct



def trainer(epochs,model,train_dataloader,val_dataloader):

    loss_track = float('inf')

    for i in range(1,epochs+1):
        TL,TA = trainFunc(i,model,train_dataloader)
        mlflow.log_metric("Training Loss", TL)
        mlflow.log_metric("Training Accuracy", TA)
        VL,VA = valFunc(i,model,train_dataloader)

        if(VL <= loss_track):
            loss_track = VL
            torch.save(model.state_dict(), state_dict_path)
            mlflow.log_artifact(state_dict_path, artifact_path="model_weights")

        mlflow.log_metric("Validation Loss", VL)
        mlflow.log_metric("Validation Accuracy", VA)
        
    
with mlflow.start_run(experiment_id=exp_id) as run:

    # Log training parameters.
    mlflow.log_params(params)
    
    trainer(params["epochs"],
            model,
            train_dataloader,
            val_dataloader)
    
    # Log model summary.
    with open("model_summary.txt", "w",encoding="utf-8") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")
    
    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "mymodel")
    
    