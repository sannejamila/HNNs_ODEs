import numpy as np
import torch
import torch.nn as nn
from NumericalIntegration.Numerical_Integration import *
from HamiltonianNeuralNetwork.HNN import *

torch.set_default_dtype(torch.float32)

#Inspo: https://github.com/shaandesai1/PortHNN/blob/main/models/TDHNN4.py

class BaseHamiltonianNeuralNetwork(nn.Module):
    def __init__(self, nstates,noutputs = 1,hidden_dim=200, act_1 = Sin(), act_2 = Sin(), act_3 =nn.Softplus()):
        super().__init__()
        self.nstates = nstates
        self.noutputs = 1
        self.hidden_dim = hidden_dim
        self.act_1 = act_1
        self.act_2 = act_2
        self.act_3 = act_3

        linear1 = nn.Linear(nstates, hidden_dim) 
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        linear3 = nn.Linear(hidden_dim, hidden_dim)
        linear4 = nn.Linear(hidden_dim, noutputs)

        for lin in [linear1, linear2, linear3,linear4]:
            nn.init.orthogonal_(lin.weight) 

        self.model = nn.Sequential(
            linear1,
            self.act_1,
            linear2,
            self.act_2,
            linear3,
            self.act_3,
            linear4,
        )

    def forward(self,u=None):
        return self.model(u)
  


class ExternalForceNeuralNetwork(nn.Module):
    def __init__(self, nstates,hidden_dim=200, act_1 = Sin(), act_2 = Sin(), act_3 =Sin()):
        super().__init__()
        self.nstates = nstates
        self.noutputs = 1
        self.hidden_dim = hidden_dim
        self.act_1 = act_1
        self.act_2 = act_2
        self.act_3 = act_3

        linear1= nn.Linear(1, hidden_dim)
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        linear3 = nn.Linear(hidden_dim, hidden_dim)
        linear4= nn.Linear(hidden_dim, int(self.nstates/2), bias=False)

        #for lin in [linear1, linear2, linear3]:
            #nn.init.orthogonal_(lin.weight) 

        self.model = nn.Sequential(
            linear1,
            self.act_1,
            linear2,
            self.act_2,
            linear3,
            self.act_3,
            linear4
        )

    def forward(self,t=None):
        return self.model(t)
    
  

class PortHamiltonianNeuralNetwork(torch.nn.Module):
    def __init__(self, nstates, S,Hamiltonian_est, External_Forces_est):
        super().__init__()
        assert nstates % 2 == 0, "Number of states must be even (q,p)"
        self.nstates = nstates
        self.npos = nstates // 2
        self.S = S
        self.External_Forces_est = External_Forces_est
        self.Hamiltonian_est = Hamiltonian_est #husk 200 hidden units nå! med 3 hidden layers
    
        #Damping: N 
        self.N = nn.Parameter(torch.zeros(1, int(self.nstates/2)))
        self.N = nn.init.kaiming_normal_(self.N)
  
    def Hamiltonian(self, u):
        return self.Hamiltonian_est(u)

    def dH(self, u):
        u = u.requires_grad_()
        return torch.autograd.grad(
            self.Hamiltonian(u).sum(),
            u,
            retain_graph=self.training,
            create_graph=self.training,
        )[0]
    
    def Get_N(self):
        return self.N
    
    def External_Force(self,t):
        return self.External_Forces_est(t)
    
    def u_dot(self,u,t):
        #Vet ikke om riktig enda
        u_dot = self.dH(u)@self.S.T

        qdot = u_dot[:,:int(self.nstates/2)]
        pdot = u_dot[:,int(self.nstates/2):]

        F = self.External_Force(t.reshape(-1, 1))

        new_pdot = pdot + self.N*qdot + F
        return torch.cat([qdot, new_pdot], 1)


