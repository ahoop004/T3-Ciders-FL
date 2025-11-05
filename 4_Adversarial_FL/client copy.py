import os
from copy import deepcopy
from math import ceil
import logging
import matplotlib.pyplot as plt
from util_functions import set_logger, save_plt
import torch
import importlib
import numpy as np
from torch.utils.data import DataLoader
from model import *
from load_data_for_clients import dist_data_per_client
from util_functions import set_seed, evaluate_fn, run_fl



class Client():
  
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.x = None
        self.y = None

    def client_update(self):

        self.y = deepcopy(self.x) 
        self.y.to(self.device)
        
        for epoch in range(self.num_epochs):

            for inputs, labels in self.data:
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                output = self.y(inputs)
                loss = self.criterion(output, labels)
                grads = torch.autograd.grad(loss, self.y.parameters(), retain_graph=False)
            
                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data -= self.lr * grad.data

            if self.device == "cuda": torch.cuda.empty_cache()