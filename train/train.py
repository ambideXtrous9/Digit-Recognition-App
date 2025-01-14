from tqdm import tqdm
import torch
from torch import nn
from model import MNISTNeuralNet
import mlflow
from torchinfo import summary
from DataLoader import train_val_dataloader
from hyperparams import params

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Digit-Recognition")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNISTNeuralNet(hidden_dim=params["hidden_dim"],dropout_prob=params["dropout_prob"])

model.to(device)

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

    for i in range(1,epochs+1):
        TL,TA = trainFunc(i,model,train_dataloader)
        mlflow.log_metric("Training Loss", TL)
        mlflow.log_metric("Training Accuracy", TA)
        VL,VA = valFunc(i,model,train_dataloader)
        mlflow.log_metric("Validation Loss", VL)
        mlflow.log_metric("Validation Accuracy", VA)
        
    
with mlflow.start_run():

    # Log training parameters.
    mlflow.log_params(params)
    
    trainer(params["epochs"],
            model,
            train_dataloader,
            val_dataloader)
    
    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")
    
    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "model")
    
    