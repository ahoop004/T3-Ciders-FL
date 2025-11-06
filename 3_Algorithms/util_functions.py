import os
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms

def set_logger(log_path):

    logger = logging.getLogger()
    logger.handlers.clear()

    logger.setLevel(logging.INFO)    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
      
def save_plt(x,y,xlabel,ylabel,filename):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def create_data(data_path, dataset_name):

    original_name = dataset_name
    key = dataset_name.upper()

    if key == "IMAGENETTE":
        from torchvision.datasets import Imagenette

        base_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        train_transform = transforms.Compose(base_transforms)
        eval_transform = transforms.Compose(base_transforms + [normalize])

        train_data = Imagenette(
            root=data_path,
            split="train",
            size="full",
            download=True,
            transform=train_transform,
        )
        test_data = Imagenette(
            root=data_path,
            split="val",
            size="full",
            download=True,
            transform=eval_transform,
        )

        imgs, labels = [], []
        for img_tensor, label in train_data:
            img = img_tensor.cpu().permute(1, 2, 0).numpy()
            imgs.append(img)
            labels.append(label)
        train_data.data = np.stack(imgs)
        train_data.targets = labels

        return train_data, test_data

    if hasattr(datasets, key):

        if key == "CIFAR10":
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616)),
            ])
        elif key == "MNIST":
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        else:

            T = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])

        DatasetClass = datasets.__dict__[key]
        params = {}
        if "train" in DatasetClass.__init__.__code__.co_varnames:
            params["train"] = True
            train_data = DatasetClass(root=data_path, download=True,
                                      transform=T, **params)
            params["train"] = False
            test_data  = DatasetClass(root=data_path, download=True,
                                      transform=T, **params)
        else:

            train_data = DatasetClass(root=data_path, split="train",
                                      download=True, transform=T)
            test_data  = DatasetClass(root=data_path, split="test",
                                      download=True, transform=T)

    else:
        raise AttributeError(
            f"...dataset \"{original_name}\" is not supported or cannot be found in TorchVision Datasets!"
        )


    if hasattr(train_data, "data") and train_data.data.ndim == 3:
        train_data.data = train_data.data.unsqueeze(3)
    if hasattr(test_data, "data") and test_data.data.ndim == 3:
        test_data.data = test_data.data.unsqueeze(3)

    return train_data, test_data

class load_data(Dataset):
    def __init__(self, x, y):
        self.length = x.shape[0]
        self.x = x.permute(0,3,1,2)
        self.y = y
        # self.image_transform = transforms.Normalize((127.5, 127.5, 127.5),(127.5, 127.5, 127.5))
        # self.image_transform = transforms.Normalize((0.1307,), (0.3081,))
        self.image_transform = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225)
        )
        
        
    def __getitem__(self, index):
        image,label = self.x[index],self.y[index]
        image = self.image_transform(image)
        return image,label
        
    def __len__(self):
        return self.length

def tensor_to_numpy(data, device):
    if device.type == "cpu":
        return data.detach().numpy()
    else:
        return data.cpu().detach().numpy()

def numpy_to_tensor(data, device, dtype="float"):
    if dtype=="float":
        return torch.tensor(data, dtype=torch.float).to(device)
    elif dtype=="long":
        return torch.tensor(data, dtype=torch.long).to(device)

def evaluate_fn(dataloader,model,loss_fn,device):

    model.eval()
    running_loss = 0
    total = 0
    correct = 0

    num_batches = 0
    with torch.no_grad():
        for images, labels in dataloader:
            output = model(images.to(device))
            loss = loss_fn(output,labels.to(device))
            running_loss += loss.item()
            total += labels.size(0)
            correct += (output.argmax(dim=1).cpu() == labels.cpu()).sum().item()
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    avg_loss = running_loss/num_batches
    acc = 100*(correct/total)
    return avg_loss,acc
