import numpy as np
import torch
from NumericalIntegration.Numerical_Integration import *


class DuffingSystem:
    def __init__(self,alpha, beta, omega, delta, gamma, seed = 123):
        self.nstates = 2
        self.S = np.array([[0., 1.],[-1.,0.]])
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.delta = delta
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.name_system = "DuffingSystem"

    def initial_condition(self):
        #Inspired by https://github.com/shaandesai1/PortHNN/blob/main/duffing_chaos_inference.ipynb
        u0 =  np.random.rand(2) * 2 - 1 #uniformly between -1, 1
        radius = np.sqrt(np.random.uniform(0.5, 1.5))  #r between 0.7071 and 1.2247
        u0 /= np.sqrt((u0 ** 2).sum()) * (radius) #scaled so u0 in interval [−1.4142,1.4142]
        #u0 = np.random.rand(2)*0,
        return u0.flatten()

    def Hamiltonian(self,u,t):
        #Includes external force
        u = u.reshape(-1)

        if u.ndim ==1:
            q,p  = u[0],u[1]
        else:
            q,p = u[:,0],u[:,1]

        H = self.alpha * (q**2)/2 + p**2/2 + self.beta*q**4/4#-q*self.gamma*np.sin(self.omega*t)
        return H
   
    
    def Hamiltonian_grad(self,u,t):
        u = u.reshape(-1)
        if u.ndim ==1:
            q,p = u[0],u[1]
        else:
            q,p = u[:,0],u[:,1]
        dHdq = np.array([self.alpha*q + self.beta*q**3]) #- self.gamma*np.sin(self.omega*t)

        dHdp = np.array([p])

        if isinstance(u, np.ndarray):
       
            dHdu = np.array([dHdq,dHdp])
        else: 
            dHdu = torch.tensor([dHdq,dHdp])
        return dHdu
    

    def u_dot(self,u,t_start):
     
        t = t_start
        if u.ndim ==1:
            q,p  = u[0],u[1]
        else:
            q,p = u[:,0],u[:,1]
        dH = self.Hamiltonian_grad(u.T,t).T
        dissipation = np.stack([np.zeros_like(p), -self.delta * p], axis=1) 
        external_force = np.stack([np.zeros_like(p), self.gamma*np.sin(self.omega*t)], axis=1) #changed to sin in stead of cos
        u_dot = dH@self.S.T + dissipation +external_force
        #u_dot = np.concatenate([p,  q - q ** 3 - self.delta * p + self.gamma * np.cos(self.omega * t)], axis=-1)
        return u_dot

    def sample_trajectory(self,t,u0=None, integrator =  "symplectic midpoint"):
        if u0 is None:
            u0 = self.initial_condition() #not implemented yet

        #Initializing solution and its derivative
        u = np.zeros([t.shape[0],self.nstates])
        dudt = np.zeros_like(u)
        
        #Setting initial conditions
        u[0, :] = u0

        for i, time_step in enumerate(t[:-1]):
            dt = t[i+1]-t[i]
        
            if integrator == "midpoint":
                dudt[i,:] = explicit_midpoint_time_derivative(self.u_dot,u_start = u[i : i +1, :],t_start = np.array([time_step]), dt = dt)
            elif integrator == "symplectic midpoint":
                dudt[i,:] = symplectic_midpoint_time_derivative(self.u_dot,u_start = u[i : i +1, :],t_start = np.array([time_step]),dt = dt)
          
            u[i+1,:] = u[i,:] + dt*dudt[i,:]

        return u, dudt, t, u0
    


