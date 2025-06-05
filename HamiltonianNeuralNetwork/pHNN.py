import numpy as np
import torch
import torch.nn as nn
from NumericalIntegration.Numerical_Integration import *
from HamiltonianNeuralNetwork.HNN import *

torch.set_default_dtype(torch.float32)

#Inspo: https://github.com/shaandesai1/PortHNN/blob/main/models/TDHNN4.py

class BaseHamiltonianNeuralNetwork(nn.Module):
    def __init__(self, nstates,noutputs = 1,hidden_dim=100, act_1 = Sin(), act_2 = Sin()):
        super().__init__()
        self.nstates = nstates
        self.noutputs = 1
        self.hidden_dim = hidden_dim
        self.act_1 = act_1
        self.act_2 = act_2
    
        linear1 = nn.Linear(nstates, hidden_dim) 
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        linear3 = nn.Linear(hidden_dim, noutputs)

        for lin in [linear1, linear2, linear3]:#,linear4]:
            nn.init.orthogonal_(lin.weight) 

        self.model = nn.Sequential(
            linear1,
            self.act_1,
            linear2,
            self.act_2,
            linear3,
        )

    def forward(self,u=None):
        return self.model(u)
  


class ExternalForceNeuralNetwork(nn.Module):
    def __init__(self, nstates,hidden_dim=100, act_1 = Sin(), act_2 = Sin()):
        super().__init__()
        self.nstates = nstates
        self.noutputs = 1
        self.hidden_dim = hidden_dim
        self.act_1 = act_1
        self.act_2 = act_2
        self.Fourier = False
        self.fourier_to_scalar = nn.Linear(2, 1)
        #self.fourier_to_scalar = nn.Sequential(
        #nn.Linear(2, 10),
        #nn.Tanh(),  
        #nn.Linear(10, 1))
        #Initializing weigths and bias Fourier Basis
        if self.Fourier:
            nn.init.kaiming_normal_(self.fourier_to_scalar.weight)
            nn.init.zeros_(self.fourier_to_scalar.bias)



        self.learned_period = None

        linear1= nn.Linear(1, hidden_dim)
        linear2 = nn.Linear(hidden_dim, hidden_dim)
        linear3= nn.Linear(hidden_dim, int(self.nstates/2), bias=False)

        for lin in [linear1, linear2, linear3]:
            nn.init.orthogonal_(lin.weight) 

        self.model = nn.Sequential(
            linear1,
            self.act_1,
            linear2,
            self.act_2,
            linear3,
        )
    """

    def init_fourier_to_scalar(self):
        for l in self.fourier_to_scalar:
            if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight)
                #nn.init.xavier_uniform_(l.weight)
                nn.init.zeros_(l.bias)

   """

    def forward(self,t=None):
        if self.Fourier:
            if t.ndim == 1: 
                t = t.unsqueeze(-1) 
            fourier_basis = torch.cat([
                    torch.sin(2 * torch.pi * t / self.learned_period),
                    torch.cos(2 * torch.pi * t / self.learned_period),
                ], dim=-1)
            t_scalar = self.fourier_to_scalar(fourier_basis) #Preprocess Fourier features back to scalar
            return self.model(t_scalar) 
        else:
            return self.model(t)
    
  

class PortHamiltonianNeuralNetwork(torch.nn.Module):
    def __init__(self, nstates, S,Hamiltonian_est, External_Forces_est):
        super().__init__()
        assert nstates % 2 == 0 #Number of states must be even for (q,p)
        self.nstates = nstates
        self.npos = nstates // 2
        self.S = torch.tensor(S,dtype=torch.float32)
        self.External_Forces_est = External_Forces_est
        self.Hamiltonian_est = Hamiltonian_est 
        self.act1 = self.Hamiltonian_est.act_1
        self.act2 = self.Hamiltonian_est.act_2
    
        #Damping N 
        self.N = nn.Parameter(torch.zeros(1, int(self.nstates/2)))
        self.N = nn.init.kaiming_normal_(self.N)
  
    def Hamiltonian(self, u):
        return self.Hamiltonian_est(u)
    
    def time_derivative_step(self, integrator, u_start, dt, u_end=None, *args, **kwargs):
        if integrator == "RK4":
            return RK4_time_derivative(self.u_dot, u_start, dt)
        elif integrator == "midpoint":
            return explicit_midpoint_time_derivative(self.u_dot, u_start, dt, *args, **kwargs)
        elif integrator == "symplectic midpoint":
            return symplectic_midpoint_time_derivative(self.u_dot, u_start, dt, u_end, *args, **kwargs)
        elif integrator == "symplectic euler":
            return symplectic_euler(self.u_dot, u_start, dt)

    def dH(self, u):
        #u = u.requires_grad_()
        u = u.detach().requires_grad_()
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
    
    def u_dot(self,u,t_start):
        t  = t_start
        u_dot = self.dH(u)@self.S.T
        if u_dot.ndim == 1:
            u_dot = u_dot.unsqueeze(0)

        #qdot = dH/dp
        #pdot = -dH/dq
        qdot = u_dot[:,:int(self.nstates/2)]
        pdot = u_dot[:,int(self.nstates/2):]

        F = self.External_Force(t.reshape(-1, 1))
        damping_term = (self.N @ qdot.T).T 

         #pdot = -dH/dq+N*dH/dp+F
        new_pdot = pdot + damping_term + F

        return torch.cat([qdot, new_pdot], 1)


    def simulate_trajectory(self,integrator,t_sample,dt,u0=None):
        if u0 is None:
            u0 = self.initial_condition_sampler()
 
        u0 = torch.tensor(u0,dtype = torch.float32)
        u0 = u0.reshape(1,u0.shape[-1])

        t_sample = torch.tensor(t_sample,dtype = torch.float32)

        #Initializing solution 
        u = torch.zeros([t_sample.shape[-1],self.nstates])
        dudt = torch.zeros_like(u)

        #Setting initial conditions
        u[0, :] = u0

        for i, t_step in enumerate(t_sample[:-1]):
            dt = t_sample[i + 1] - t_step
            dudt[i,:] = self.time_derivative_step(integrator=integrator,u_start = u[i : i + 1, :],t_start = t_step, dt = dt)
            u[i+1,:] = u[i,:] + dt*dudt[i,:]
        return u,dudt,u0
    
    def generate_trajectories(self,ntrajectories, t_sample,integrator = "midpoint",u0s=None):
        if u0s.any() == None:
            u0s = self.initial_condition_sampler(ntrajectories)
        
        #Reshaping
        u0s = torch.tensor(u0s,dtype = torch.float32)
        u0s = u0s.reshape(ntrajectories, self.nstates)
        t_sample = torch.tensor(t_sample,dtype = torch.float32)
        if len(t_sample.shape) == 1:
                #Reshaping time
                t_sample = np.tile(t_sample, (ntrajectories, 1))

        dt = t_sample[0, 1] - t_sample[0, 0]
        traj_length = t_sample.shape[-1]

        #Initializng u and setting initial conditions
        u = torch.zeros([ntrajectories, traj_length, self.nstates])
        u[:,0,:] = u0s

        for i in range(ntrajectories):
            u[i] = self.simulate_trajectory(integrator = integrator,t_sample = t_sample[i], u0 = u0s[i],dt=dt)[0]
   
        return u, t_sample