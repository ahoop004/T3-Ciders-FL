import torch
import torch.nn as nn
from copy import deepcopy
from model import MobileNetV2Transfer, MobileNetV3Transfer
from client import Client                         
from util_functions import set_seed
from attacks import fgsm, rand_noise_attack, pgd_attack



class MaliciousClient(Client):
    def __init__(self, client_id, local_data, device,
                 num_epochs, criterion, lr, attack_config):
        super().__init__(client_id, local_data, device,
                         num_epochs, criterion, lr)
        self.attack_config = attack_config
        self.attack_type   = attack_config.get("type", "pgd")
        self.device=device
        self.surrogate = MobileNetV2Transfer(
            pretrained=attack_config["surrogate_pretrained"],
            num_classes=attack_config.get("num_classes", 10)
        ).to(self.device)

        #
        self.opt = torch.optim.Adam(
            self.surrogate.parameters(),
            lr=attack_config["surrogate_lr"]
        )
        self.criterion = eval(attack_config["criterion"])()
        self.ft_epochs = attack_config["surrogate_finetune_epochs"]
        self.batch_size = attack_config["surrogate_batch_size"]

    def train_surrogate(self, train_loader):
        self.surrogate.train()
        for _ in range(self.ft_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                preds = self.surrogate(x)
                loss = self.criterion(preds, y)
                loss.backward()
                self.opt.step()


    def perform_attack(self, x, y):

        attack_type = self.attack_config.get("type", "pgd")

        if attack_type == "fgsm":
            # Fast Gradient Sign Method
            return fgsm(
                model     = self.surrogate,
                criterion = self.criterion,
                images    = x,
                labels    = y,
                step_size = self.attack_config["step_size"]
            )

        elif attack_type == "rand_noise":
            # Random noise injection
            return rand_noise_attack(
                images    = x,
                step_size = self.attack_config["step_size"]
            )

        elif attack_type == "pgd":
            # Projected Gradient Descent
            return pgd_attack(
                model     = self.surrogate,
                criterion = self.criterion,
                images    = x,
                labels    = y,
                eps       = self.attack_config["epsilon"],
                step_size = self.attack_config["step_size"],
                iters     = self.attack_config["iters"],
            )

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    def client_update(self):
        
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        poison_rate = self.attack_config.get("poison_rate", 0.0)
        target_lbl  = self.attack_config["target_label"]

        for epoch in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                mask = torch.rand(labels.size(0), device=self.device) < poison_rate
                if mask.any():
  
                    clean_inputs = inputs[mask]
 
                    poison_labels = torch.full(
                        (mask.sum().item(),),
                        target_lbl,
                        dtype=labels.dtype,
                        device=self.device
                    )

                    adv_examples = self.perform_attack(clean_inputs, poison_labels)

                    inputs[mask] = adv_examples
                    labels[mask] = poison_labels


                outputs = self.y(inputs)
                loss    = self.criterion(outputs, labels)
                grads   = torch.autograd.grad(loss, self.y.parameters())

                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data -= self.lr * grad.data