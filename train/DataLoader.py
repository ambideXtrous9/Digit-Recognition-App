import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(15),    # Randomly rotate the image by a certain degree
    transforms.RandomAffine(degrees=0, shear=10),  # Apply random shear transformation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly change brightness, contrast, and saturation
    transforms.ToTensor(),            # Convert the image to a tensor
])

train = torchvision.datasets.MNIST(root='./',
                                   download=True,
                                   transform=transform,
                                   train=True)

test = torchvision.datasets.MNIST(root='./',
                                   download=True,
                                   transform=transform,
                                   train=False)


def train_val_dataloader(batch_size = 32):

    train_dataloader = DataLoader(train,batch_size = batch_size,shuffle=True,drop_last=True)
    val_dataloader = DataLoader(test,batch_size = batch_size,shuffle=False,drop_last=True)
    
    return train_dataloader, val_dataloader