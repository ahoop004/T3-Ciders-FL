import os
from copy import deepcopy
import logging
from util_functions import set_logger, save_plt
import torch
import importlib
import numpy as np
from torch.utils.data import DataLoader
from model import *
from load_data_for_clients import dist_data_per_client
from util_functions import set_seed, evaluate_fn, run_fl
from client import Client


        


class Server():

    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={},attack_config=None):
      
        set_seed(global_config["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_config["dataset_path"]
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per = data_config["non_iid_per"]
        self.fraction = fed_config["fraction_clients"]
        self.num_clients = fed_config["num_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.attack_config = attack_config
        self.criterion = eval(fed_config["criterion"])()
        self.lr = fed_config["global_stepsize"]
        self.lr_l = fed_config["local_stepsize"]
        self.model_config = model_config
        self.model_module = importlib.import_module(model_config["module"])
        self.model_class = getattr(self.model_module, model_config["name"])
        self.x = self.model_class(**model_config.get("kwargs", {}))   
        self.clients = None       
    
    def create_clients(self, local_datasets):

        from malicious_client import MaliciousClient
        m_frac = self.attack_config.get("malicious_fraction", 0)
        num_mal = int(self.num_clients * m_frac)
        mal_ids = set(np.random.choice(self.num_clients, num_mal, replace=False))
        clients = []
        for idx, dataset in enumerate(local_datasets):
            if idx in mal_ids:
                cl = MaliciousClient(
                    client_id=idx,
                    local_data=dataset,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l,
                    attack_config=self.attack_config
                )
            else:
                cl = Client(
                    client_id=idx,
                    local_data=dataset,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l
                )
            clients.append(cl)
        return clients
    
    def setup(self, **init_kwargs):
        local_datasets,test_dataset = dist_data_per_client(self.data_path, self.dataset_name, self.num_clients, self.batch_size, self.non_iid_per, self.device)
        self.data = test_dataset
      
        self.clients = self.create_clients(local_datasets)
        logging.info("\nClients are successfully initialized")
      
    def sample_clients(self):
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_ids = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, 
        replace=False).tolist())
        return sampled_client_ids

    def communicate(self, client_ids):
        for idx in client_ids:
        
            cl_model = self.model_class(**self.model_config.get("kwargs", {})).to(self.device)

            cl_model.load_state_dict(self.x.state_dict())

            self.clients[idx].x = cl_model
               
    def update_clients(self, client_ids):
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids):
        num_participants = len(client_ids)
        if num_participants == 0:
            return

        self.x.to(self.device)
        avg_y = [torch.zeros_like(param, device=self.device) for param in self.x.parameters()]

        with torch.no_grad():
            for idx in client_ids:
                for a_y, y in zip(avg_y, self.clients[idx].y.parameters()):
                    a_y.data.add_(y.data / num_participants)

            for param, a_y in zip(self.x.parameters(), avg_y):
                param.data = a_y.data

    def step(self):
        sampled_client_ids = self.sample_clients()
        self.communicate(sampled_client_ids)
        self.update_clients(sampled_client_ids)
        logging.info("\tclient_update has completed") 
        self.server_update(sampled_client_ids)
        logging.info("\tserver_update has completed")

    def train(self):
        self.results = {"loss": [], "accuracy": []}
        for round in range(self.num_rounds):
            logging.info(f"\nCommunication Round:{round+1}")
            self.step()
            test_loss, test_acc = evaluate_fn(self.data,self.x,self.criterion,self.device)
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_acc)
            logging.info(f"\tLoss:{test_loss:.4f}   Accuracy:{test_acc:.2f}%")
            print(f"\tServer Loss:{test_loss:.4f}   Accuracy:{test_acc:.2f}%")


class FedOptClient(Client):
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        super().__init__(client_id, local_data, device, num_epochs, criterion, lr)
        self.delta_y = None


    def client_update(self):
        self.x.to(self.device)
        self.y = deepcopy(self.x) 
        self.y.to(self.device)

        for epoch in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                output = self.y(inputs)
                loss = self.criterion(output, labels) 
    
    
                grads = torch.autograd.grad(loss,self.y.parameters())
    
                with torch.no_grad():
                    for param,grad in zip(self.y.parameters(),grads):
                        param.data = param.data - self.lr * grad.data

            if self.device == "cuda": torch.cuda.empty_cache()

        with torch.no_grad():
            delta_y = [torch.zeros_like(param, device=self.device) for param in self.y.parameters()]

            for del_y, param_y, param_x in zip(delta_y, self.y.parameters(), self.x.parameters()):
                del_y.data += param_y.data.detach() - param_x.data.detach()

        self.delta_y = delta_y






class FedAdamServer(Server):
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}):
        super().__init__(model_config, global_config, data_config, fed_config, optim_config)
        
        self.m = [torch.zeros_like(param, device=self.device) for param in self.x.parameters()]
        self.v = [torch.zeros_like(param, device=self.device) for param in self.x.parameters()]
        self.beta1 = optim_config.get("beta1", 0.9)
        self.beta2 = optim_config.get("beta2", 0.99)
        self.epsilon = optim_config.get("epsilon", 1e-6)
        self.timestep = 1



    def create_clients(self, local_datasets):
        clients = []
        for id_num, dataset in enumerate(local_datasets):
            client = FedOptClient(
                client_id=id_num,
                local_data=dataset,
                device=self.device,
                num_epochs=self.num_epochs,
                criterion=self.criterion,
                lr=self.lr_l
            )
            clients.append(client)
        return clients

    def server_update(self, client_ids):
        num_participants = len(client_ids)
        if num_participants == 0:
            return

        self.x.to(self.device)
        gradients = [torch.zeros_like(param, device=self.device) for param in self.x.parameters()]
        with torch.no_grad():
            for idx in client_ids:
                for grad, diff in zip(gradients, self.clients[idx].delta_y):
                    grad.data.add_(diff.data / num_participants)

            for p, g, m, v in zip(self.x.parameters(), gradients, self.m, self.v):
                m.data = self.beta1 * m.data + (1 - self.beta1) * g.data
                v.data = self.beta2 * v.data + (1 - self.beta2) * torch.square(g.data)
                m_bias_corr = m / (1 - self.beta1 ** self.timestep)
                v_bias_corr = v / (1 - self.beta2 ** self.timestep)
                p.data += self.lr * m_bias_corr / (torch.sqrt(v_bias_corr) + self.epsilon)

        self.timestep += 1

class FedAdagradServer(Server):
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}):
        super().__init__(model_config, global_config, data_config, fed_config, optim_config)

        self.s = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()] #state
        self.epsilon = 1e-6



    def create_clients(self, local_datasets):
        clients = []
        for id_num, dataset in enumerate(local_datasets):
            client = FedOptClient(
                client_id=id_num,
                local_data=dataset,
                device=self.device,
                num_epochs=self.num_epochs,
                criterion=self.criterion,
                lr=self.lr_l
            )
            clients.append(client)
        return clients

    def server_update(self, client_ids):
        num_participants = len(client_ids)
        if num_participants == 0:
            return

        self.x.to(self.device)
        gradients = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()] #gradients or delta_x
        with torch.no_grad():
            for idx in client_ids:
                for grad, diff in zip(gradients, self.clients[idx].delta_y):
                    grad.data.add_(diff.data / num_participants)

            for p,g,s in zip(self.x.parameters(), gradients, self.s):
                s.data += torch.square(g.data)
                p.data += self.lr * g.data / torch.sqrt(s.data + self.epsilon)



class FedYogiServer(Server):
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}):
        super().__init__(model_config, global_config, data_config, fed_config, optim_config)

        self.m = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()] 
        self.v = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.timestep = 1



    def create_clients(self, local_datasets):
        clients = []
        for id_num, dataset in enumerate(local_datasets):
            client = FedOptClient(
                client_id=id_num,
                local_data=dataset,
                device=self.device,
                num_epochs=self.num_epochs,
                criterion=self.criterion,
                lr=self.lr_l
            )
            clients.append(client)
        return clients

    def server_update(self, client_ids):
        num_participants = len(client_ids)
        if num_participants == 0:
            return

        self.x.to(self.device)
        gradients = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()] 
        with torch.no_grad():
            for idx in client_ids:
       
                for grad, diff in zip(gradients, self.clients[idx].delta_y):
                    grad.data.add_(diff.data / num_participants)

            for p,g,m,v in zip(self.x.parameters(), gradients, self.m, self.v):
                m.data = self.beta1 * m.data + (1 - self.beta1) * g.data
                v.data = v.data + (1 - self.beta2) * torch.sign( torch.square(g.data) - v.data) * torch.square(g.data)
                m_bias_corr = m / (1 - self.beta1 ** self.timestep)
                v_bias_corr = v / (1 - self.beta2 ** self.timestep) 
                p.data += self.lr * m_bias_corr / (torch.sqrt(v_bias_corr) + self.epsilon)

        self.timestep += 1


class ScaffoldClient(Client):
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr, client_c):
        super().__init__(client_id, local_data, device, num_epochs, criterion, lr)
        self.server_c = None
        self.client_c = client_c
        self.delta_y = None
        self.delta_c = None

    def client_update(self):
        self.x.to(self.device)
        self.y = deepcopy(self.x) 
        self.y.to(self.device)

        for inputs, labels in self.data:
            inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
            output = self.y(inputs)
            loss = self.criterion(output, labels) 
            grads = torch.autograd.grad(loss,self.y.parameters())

            with torch.no_grad():
                for param,grad,s_c,c_c in zip(self.y.parameters(),grads,self.server_c,self.client_c):
                    s_c, c_c = s_c.to(self.device), c_c.to(self.device)
                    param.data = param.data - self.lr * (grad.data + (s_c.data - c_c.data))

            if self.device == "cuda": torch.cuda.empty_cache()

        with torch.no_grad():
            delta_y = [torch.zeros_like(param, device=self.device) for param in self.y.parameters()]
            delta_c = deepcopy(delta_y)
            new_client_c = deepcopy(delta_y)

            for del_y, param_y, param_x in zip(delta_y, self.y.parameters(), self.x.parameters()):
                del_y.data += param_y.data.detach() - param_x.data.detach()
            a = (ceil(len(self.data.dataset) / self.data.batch_size)*self.num_epochs*self.lr)
            for n_c, c_l, c_g, diff in zip(new_client_c, self.client_c, self.server_c, delta_y):
                n_c.data += c_l.data - c_g.data - diff.data / a

            for d_c, n_c_l, c_l in zip(delta_c, new_client_c, self.client_c):
                d_c.data.add_(n_c_l.data - c_l.data)

        self.client_c = deepcopy(new_client_c)
        self.delta_y = delta_y
        self.delta_c = delta_c

class ScaffoldServer(Server):
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}):
        super().__init__(model_config, global_config, data_config, fed_config, optim_config)
        self.server_c = [torch.zeros_like(param,device=self.device) for param in self.x.parameters()]
        self.c_init = optim_config.get("c_init", 0.0)

    def create_clients(self, local_datasets):
        clients = []
        for id_num, dataset in enumerate(local_datasets):
            client_c = [torch.full_like(param, self.c_init, device=self.device) for param in self.x.parameters()]
            client = ScaffoldClient(
                client_id=id_num,
                local_data=dataset,
                device=self.device,
                num_epochs=self.num_epochs,
                criterion=self.criterion,
                lr=self.lr_l,
                client_c=client_c
            )
            clients.append(client)
        return clients

    def communicate(self, client_ids):
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.x)
            self.clients[idx].server_c = deepcopy(self.server_c)

    def server_update(self, client_ids):
        self.x.to(self.device)
        num_participants = len(client_ids)
        if num_participants == 0:
            return
        for idx in client_ids:
            with torch.no_grad():
                for param, diff in zip(self.x.parameters(), self.clients[idx].delta_y):
                    param.data.add_(diff.data * self.lr / num_participants)
                for c_g, c_d in zip(self.server_c, self.clients[idx].delta_c):
                    c_g.data.add_(c_d.data * self.fraction)
