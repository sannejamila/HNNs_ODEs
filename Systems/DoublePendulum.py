import numpy as np
import torch
from NumericalIntegration.Numerical_Integration import *


class DoublePendulum:
    def __init__(self,seed = 123):
        self.nstates = 4
        self.S = np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],[-1.,0.,0.,0.],[0.,-1.,0.,0.]])
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.name_system = "DoublePendulum"


    def Hamiltonian(self,u):
        u = u.reshape(-1)

        if u.ndim ==1:
            x,y,px,py = u[0],u[1],u[2],u[3]
        else:
            x,y,px,py = u[:,0],u[:,1],u[:,2],u[:,3]

        H = (px**2+2*py**2-2*px*py*np.cos(x-y))/(2*(1+np.sin(x-y)**2))-2*np.cos(x)-np.cos(y)
        return H
   
    
    def Hamiltonian_grad(self,u):

        u = u.reshape(-1)
   
        if u.ndim ==1:
            x,y,px,py = u[0],u[1],u[2],u[3]
        else:
            x,y,px,py = u[:,0],u[:,1],u[:,2],u[:,3]
        h1 = px*py*np.sin(x-y)/(1+np.sin(x-y)**2)
        h2 = 1/2*(px**2+2*py**2-2*px*py*np.cos(x-y))/(1+np.sin(x-y)**2)**2

        dHdx = 2*np.sin(x)+h1-h2*np.sin(2*(x-y))
        dHdy = np.sin(y)-h1+h2*np.sin(2*(x-y))
        dHdpx = (px-py*np.cos(x-y))/(1+np.sin(x-y)**2)
        dHdpy = (-px*np.cos(x-y)+2*py)/(1+np.sin(x-y)**2)

        if isinstance(u, np.ndarray):
            dHdu = np.array([dHdx,dHdy ,dHdpx,dHdpy])
        else: 
            dHdu = torch.tensor([dHdx,dHdy ,dHdpx,dHdpy])
        return dHdu
    
    
        
    
    def initial_condition(self,H0=None):
        min_norm=0.3
        max_norm=0.6
        dim=4
        while True:
            vec = np.random.randn(dim)  # Generate random vector from normal distribution
            norm = np.linalg.norm(vec)  # Compute L2 norm
            if norm == 0:
                continue  # Avoid division by zero
            else:
                break

        vec = vec / norm  # Normalize to unit vector
        scale = np.random.uniform(min_norm, max_norm)
        vec = vec * scale
        x0, y0, px0, py0 = vec[0], vec[1], vec[2], vec[3]
        return np.array([x0,y0,px0,py0]).flatten()
  


    def u_dot(self,u):
        dH = self.Hamiltonian_grad(u.T).T
        u_dot = dH@self.S.T
        return u_dot

    
    def sample_trajectory(self,t,u0= None,H0 = None, integrator = "symplectic midpoint"):
        if u0 is None:
            u0 = self.initial_condition(H0)

        #Initializing solution and its derivative
        u = np.zeros([t.shape[0],self.nstates])
        dudt = np.zeros_like(u)
        
        #Setting initial conditions
        u[0, :] = u0

        for i, time_step in enumerate(t[:-1]):
            dt = t[i+1]-t[i]
            if integrator == "RK4":
                dudt[i,:] = RK4_time_derivative(self.u_dot,u_start = u[i : i + 1, :], dt = dt)
            elif integrator == "midpoint":
                dudt[i,:] = explicit_midpoint_time_derivative(self.u_dot,u_start = u[i : i +1, :], dt = dt)
            elif integrator == "symplectic midpoint":
                dudt[i,:] = symplectic_midpoint_time_derivative(self.u_dot,u_start = u[i : i +1, :],dt = dt)
            elif integrator == "symplectic euler":
                dudt[i,:] = symplectic_euler(self.u_dot,u_start = u[i : i +1, :],dt = dt)

            u[i+1,:] = u[i,:] + dt*dudt[i,:]

        return u, dudt, t, u0

class DoublePendulumExternalForce(DoublePendulum):
    def __init__(self, seed=123):
        super().__init__(seed=seed)  
        self.name_system = "DoublePendulumExternalForce"

    
    def External_force(self, t):
        F_x = 0.5 * np.cos(0.7 * np.pi * t)
        F_y = 0.2 * np.sin(1.3 * np.pi * t)
        if isinstance(t, np.ndarray):
            F = np.stack([F_x, F_y], axis=-1)
        else:
            F = torch.stack([F_x, F_y], dim=-1)
        return F  # shape: [batch_size, 2] or [2] if scalar

    def u_dot(self,u,t_start):
        t = t_start
        dH = self.Hamiltonian_grad(u.T).T
        #print("shape dh in udot: ",dH.shape)
     
   
        
        dissipation = np.zeros_like(u)
        if dH.ndim ==1:
      
            dissipation[:,2] = - 0.1* dH[2]
            dissipation[:,3] = -0.05 * dH[3]
        else:
            dissipation[:,2] = - 0.1* dH[:,2]
            dissipation[:,3] = -0.05 * dH[:,3]

        external_force = np.zeros_like(u)
        #external_force[:,2] = 0.5 * np.cos(0.7 * np.pi * t)
        #external_force[:,3] =  0.2 * np.sin(1.3 * np.pi * t)

        #dissipation = self.get_dissipation(u=u,dH=dH)

        F = self.External_force(t)

        if F.ndim == 1:
            F = F.reshape(1, 2)  # make it [1, 2] for broadcasting

        external_force[:, 2:] = F  # apply external force to px and py


        u_dot = dH@self.S.T + dissipation +external_force
        return u_dot
    """

    def get_dissipation(self,u,dH=None):
        if dH is None:
            dH = self.Hamiltonian_grad(u.T).T
        dissipation = np.zeros_like(u)
        if dH.ndim ==1:
            #print("shape dh in diss: ",dH.shape)
            dissipation[:,2] = - 0.1* dH[2]
            dissipation[:,3] = -0.05 * dH[3]
        else:
            dissipation[:,2] = - 0.1* dH[:,2]
            dissipation[:,3] = -0.05 * dH[:,3]
        return dissipation
"""
    def get_dissipation(self):
        return np.array([[ -0.1, -0.05]])

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

"""



class DoublePendulumExternalForce:
    def __init__(self,seed = 123):
        self.nstates = 4
        self.S = np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],[-1.,0.,0.,0.],[0.,-1.,0.,0.]])
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.name_system = "DoublePendulumExternalForce"


    def Hamiltonian(self,u):
        u = u.reshape(-1)

        if u.ndim ==1:
            x,y,px,py = u[0],u[1],u[2],u[3]
        else:
            x,y,px,py = u[:,0],u[:,1],u[:,2],u[:,3]

        H = (px**2+2*py**2-2*px*py*np.cos(x-y))/(2*(1+np.sin(x-y)**2))-2*np.cos(x)-np.cos(y)
        return H
   
    
    def Hamiltonian_grad(self,u):

        u = u.reshape(-1)
   
        if u.ndim ==1:
            x,y,px,py = u[0],u[1],u[2],u[3]
        else:
            x,y,px,py = u[:,0],u[:,1],u[:,2],u[:,3]
        h1 = px*py*np.sin(x-y)/(1+np.sin(x-y)**2)
        h2 = 1/2*(px**2+2*py**2-2*px*py*np.cos(x-y))/(1+np.sin(x-y)**2)**2

        dHdx = 2*np.sin(x)+h1-h2*np.sin(2*(x-y))
        dHdy = np.sin(y)-h1+h2*np.sin(2*(x-y))
        dHdpx = (px-py*np.cos(x-y))/(1+np.sin(x-y)**2)
        dHdpy = (-px*np.cos(x-y)+2*py)/(1+np.sin(x-y)**2)

        if isinstance(u, np.ndarray):
            dHdu = np.array([dHdx,dHdy ,dHdpx,dHdpy])
        else: 
            dHdu = torch.tensor([dHdx,dHdy ,dHdpx,dHdpy])
        return dHdu
    
    
    def External_force(self, t):
        F_x = 0.5 * np.cos(0.7 * np.pi * t)
        F_y = 0.2 * np.sin(1.3 * np.pi * t)
        if isinstance(t, np.ndarray):
            F = np.stack([F_x, F_y], axis=-1)
        else:
            F = torch.stack([F_x, F_y], dim=-1)
        return F  # shape: [batch_size, 2] or [2] if scalar

  
    
    def initial_condition(self,H0=None):
        min_norm=0.3
        max_norm=0.6
        dim=4
        while True:
            vec = np.random.randn(dim)  # Generate random vector from normal distribution
            norm = np.linalg.norm(vec)  # Compute L2 norm
            if norm == 0:
                continue  # Avoid division by zero
            else:
                break

        vec = vec / norm  # Normalize to unit vector
        scale = np.random.uniform(min_norm, max_norm)
        vec = vec * scale
        x0, y0, px0, py0 = vec[0], vec[1], vec[2], vec[3]
        return np.array([x0,y0,px0,py0]).flatten()
  


    def u_dot(self,u,t_start):
        t = t_start

        dH = self.Hamiltonian_grad(u.T).T

        dissipation = np.zeros_like(u)
   
        
        if dH.ndim == 1:
            dissipation[:,2] = - 0.1* dH[2]
            dissipation[:,3] = -0.05 * dH[3]
        else:
            dissipation[:,2] = - 0.1* dH[:,2]
            dissipation[:,3] = -0.05 * dH[:,3]


        external_force = np.zeros_like(u)
        #external_force[:,2] = 0.5 * np.cos(0.7 * np.pi * t)
        #external_force[:,3] =  0.2 * np.sin(1.3 * np.pi * t)



        F = self.External_force(t)

        if F.ndim == 1:
            F = F.reshape(1, 2)  # make it [1, 2] for broadcasting

        external_force[:, 2:] = F  # apply external force to px and py


        u_dot = dH@self.S.T + dissipation +external_force
        return u_dot

    def get_dissipation(self,u):
        dH = self.Hamiltonian_grad(u.T).T
        dissipation = np.zeros_like(u)
        if dH.ndim ==1:
            dissipation[2] = - 0.1* dH[2]
            dissipation[3] = -0.05 * dH[3]
        else:
            dissipation[:,2] = - 0.1* dH[:,2]
            dissipation[:,3] = -0.05 * dH[:,3]
        return dissipation



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
"""