import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
import datetime
from torch.utils.data import Dataset, DataLoader
import inspect
#torch.set_default_device("mps")
#mps_device = torch.device("mps")
#mps_device = torch.device("mps")


class ODEDataset(Dataset):
    def __init__(self, data):
        self.inputs = data[0]
        self.targets = data[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = tuple(inp[idx] for inp in self.inputs)
        y = self.targets[idx]
        return x, y



class Training():
    def __init__(self,model,integrator,train_data,val_data,optimizer,system,batch_size,epochs,shuffle = True, verbose=True,L_coeff=None,num_workers=0):
        self.model = model
        self.integrator = integrator
        self.system = system
        self.batch_size = batch_size
        self.epochs = epochs
        self.L_coeff = L_coeff
        self.train_data = train_data
        self.val_data = val_data
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer
        self.shuffle = shuffle
        self.verbose = verbose
        self.num_workers = num_workers
        self.act1 = model.act1
        self.act2 = model.act2
        self.shape_data = train_data[0][0].shape
    
    def to_dataset(self,data,shuffle):
        dataset = ODEDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return loader

    def Batch_Data(self,data,shuffle):
        nsamples = data[1].shape[0]
        if shuffle:
            permutation = torch.randperm(nsamples)
        else:
            permutation = torch.arange(nsamples)
        nbatches = np.ceil(nsamples/self.batch_size).astype(int)
        batched = [(None,None)] *nbatches  
        for i in range(nbatches):
            indices = permutation[i * self.batch_size : (i + 1) * self.batch_size]
            input_tuple = [data[0][j][indices] for j in range(len(data[0]))]
            dudt = data[1][indices]
            batched[i] = (input_tuple, dudt)
        return batched
    """
    def train_one_epoch(self,model,batched_train_data,loss_func,optimizer):
        computed_loss = 0.0
        optimizer.zero_grad()
        for i, (input_tuple, dudt) in enumerate(batched_train_data):
            u_start, u_end, dt= input_tuple
            n,m = u_start.shape
            if n ==1:
                u_start = u_start.view(-1)
            dudt = dudt.view(n,m)
            dudt_est = model.time_derivative_step(integrator = self.integrator, u_start = u_start,u_end = u_end,dt = dt)
            loss = loss_func(dudt,dudt_est,u_start,self.system,self.L_coeff)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            computed_loss += loss.item()

        return computed_loss / len(batched_train_data)
    """
    def train_one_epoch(self, model, batched_train_data, loss_func, optimizer, penalty_func=None):
        computed_loss = 0.0
    
        for input_tuple, dudt in batched_train_data:
            optimizer.zero_grad()
            # Handle 3 or 4 input components (w/ or w/o t_start)
            if len(input_tuple) == 3:
                u_start, u_end, dt = input_tuple
                #u_start, u_end, dt = u_start.to(mps_device), u_end.to(mps_device), dt.to(mps_device) 
                t_args = ()
            elif len(input_tuple) == 4:
                u_start, u_end, t_start, dt = input_tuple
                #u_start, u_end, t_start, dt = u_start.to(mps_device), u_end.to(mps_device), t_start.to(mps_device), dt.to(mps_device) 
                #t_args = (t_start,)
                t_args = (t_start.detach(),)
                
            n, m = u_start.shape
            if n == 1:
                u_start = u_start.view(-1)
            dudt = dudt.view(n, m)#.to(mps_device)
            dudt_est = model.time_derivative_step(self.integrator,u_start,dt,u_end,*t_args)
            # Optional penalty
            loss = loss_func(dudt,dudt_est,u_start,self.system,self.L_coeff)
            if penalty_func is not None:
                penalty = penalty_func(model, *t_args)
                loss = loss + penalty 
 
            loss.backward()
            optimizer.step()
  
            computed_loss += loss.detach().item()

        return computed_loss / len(batched_train_data)

    def compute_validation_loss(self,model,valdata_batched, loss_func,penalty_func=None):
        model.eval()
        val_loss = 0
        
     
        for i, (input_tuple, dudt) in enumerate(valdata_batched):
            if len(input_tuple) == 3:
                u_start, u_end, dt = input_tuple
                #u_start, u_end, dt = u_start.to(mps_device), u_end.to(mps_device), dt.to(mps_device) 
                t_args = ()
            elif len(input_tuple) == 4:
                u_start, u_end, t_start, dt = input_tuple
                #u_start, u_end, t_start, dt = u_start.to(mps_device), u_end.to(mps_device), t_start.to(mps_device), dt.to(mps_device) 
                t_args = (t_start,)

            #u_start, u_end, dt = input_tuple
            n,m = u_start.shape
            if n ==1:
                u_start = u_start.view(-1)
            dudt = dudt.view(n,m)#.to(mps_device) 
            with torch.enable_grad():
                dudt_est = model.time_derivative_step(self.integrator,u_start,dt,u_end, *t_args)

            val_loss += loss_func(dudt,dudt_est,u_start,self.system,lam=self.L_coeff).item()
            if penalty_func is not None:
                val_loss += penalty_func(model, *t_args)
    
        val_loss = val_loss / len(valdata_batched)

        return val_loss


    def save_model(self, model, loss, val_loss):
        if self.L_coeff == None:
            path = f"Models/{self.system.name_system}/{self.system.name_system}_{self.integrator}_{self.epochs}epoch_{self.act1}_{self.act2}_batchsize_{self.batch_size}_shape_{self.shape_data}.pt"
        else:
            path = f"Models/{self.system.name_system}/{self.system.name_system}_{self.integrator}_{self.epochs}epoch_{self.act1}_{self.act2}_batchsize_{self.batch_size}_shape_{self.shape_data}_L_coeff_{self.L_coeff}.pt"
        torch.save({
                'epoch': self.epochs,
                'model': model,
                'loss': loss,
                'val_loss': val_loss,
                }, path)
        return "Model saved"
    
    def train(self,loss_func, penalty_func=None):
        trainingdetails={}
        optimizer = self.optimizer
        model = self.model
        loss_list = []
        val_loss_list = []

        train_loader, val_loader = self.to_dataset(self.train_data,self.shuffle), self.to_dataset(self.val_data, shuffle = False)


        with trange(self.epochs) as steps:
            for epoch in steps:
                train_batch = train_loader
                self.model.train(True) 
                start = datetime.datetime.now() 
                avg_loss = self.train_one_epoch(model,train_batch,loss_func,optimizer,penalty_func)
                end = datetime.datetime.now() 
                loss_list.append(avg_loss)
                self.model.train(False) 
                if self.verbose:
                    steps.set_postfix(epoch=epoch, loss=avg_loss)

                if self.val_data is not None:
                    start = datetime.datetime.now()
                    vloss = self.compute_validation_loss(model,val_loader, loss_func,penalty_func)
                    end = datetime.datetime.now()
                    val_loss_list.append(vloss)
                trainingdetails["epochs"] = epoch + 1
                trainingdetails["val_loss"] = vloss
                trainingdetails["train_loss"] = avg_loss

         # Plot the loss curve
        plt.figure(figsize=(7, 4))
        plt.plot(loss_list, label = "Training Loss")
        plt.plot(val_loss_list,label = "Validation Loss")
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

        self.trainingdetails = trainingdetails
        self.save_model(model, loss_list,val_loss_list)

        return model,trainingdetails




def loss_wrapper(loss_func, *default_args, **default_kwargs):
    sig = inspect.signature(loss_func)
    param_names = list(sig.parameters)

    def wrapped_loss(*args, **kwargs):
        all_kwargs = {**dict(zip(param_names, args)), **kwargs, **dict(zip(param_names[len(args):], default_args))}
        needed_args = {k: v for k, v in all_kwargs.items() if k in param_names}
        return loss_func(**needed_args)
    
    return wrapped_loss

def load_model(path):
    checkpoint = torch.load(path,weights_only=False)
    model = checkpoint['model'] 
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, epoch, loss