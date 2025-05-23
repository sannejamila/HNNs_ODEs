import numpy as np
import torch
from NumericalIntegration.Numerical_Integration import *

class HenonHeilesSystem:
    def __init__(self,lam = 1, seed = 123):
        self.nstates = 4
        self.lam = lam
        self.S = np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],[-1.,0.,0.,0.],[0.,-1.,0.,0.]])
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.name_system = "HenonHeiles"


    def Hamiltonian(self,u):
     
        A = np.identity(self.nstates)
        B = np.array([[0,1,0,0],[0,-1/3,0,0],[0,0,0,0],[0,0,0,0]])

        if not isinstance(u, np.ndarray):
            if isinstance(A, np.ndarray):
                A = torch.tensor(A)
            if isinstance(B, np.ndarray):
                B = torch.tensor(B)
            A = A.to(u.dtype)
            B = B.to(u.dtype)

        H = 1/2*u.T@A@u + self.lam*(u**2).T@B@u
        return H
   
    
    def Hamiltonian_grad(self,u):
        lam = self.lam
        u = u.reshape(-1)

        if u.ndim ==1:
            x,y,px,py = u[0],u[1],u[2],u[3]
        else:
            x,y,px,py = u[:,0],u[:,1],u[:,2],u[:,3]
        if isinstance(u, np.ndarray):
            dHdu = np.array([x+2*lam*x*y,y+lam*(x**2-y**2),px,py])
        else: 
            dHdu = torch.tensor([x+2*lam*x*y,y+lam*(x**2-y**2),px,py])
        return dHdu
    

        
    
    def initial_condition(self, H0=None):
        x0= 0
        if H0 is None:
            H0 = np.random.uniform(0,1/10)
        while True:
            y0 = np.random.uniform(-1,1)    
            py0 = np.random.uniform(-1,1)
            K = 2*H0-(py0**2 +y0**2-self.lam/3*y0**3)
            if K>= 0:
                break
        px0 = np.sqrt(K)
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
    

class HenonHeilesExternalForce(HenonHeilesSystem):
    def __init__(self, seed=123):
        super().__init__(seed=seed) 
        self.name_system = "HenonHeilesExternalForce"

    
    def External_force(self, t):
        F_x = 0.8 * np.cos(0.8 * np.pi * t)
        F_y = 0.35 * np.sin(1.1 * np.pi * t)
        if isinstance(t, np.ndarray):
            F = np.stack([F_x, F_y], axis=-1)
        else:
            F = torch.stack([F_x, F_y], dim=-1)
        return F  

    def u_dot(self,u,t_start):
        t = t_start
        dH = self.Hamiltonian_grad(u.T).T
        external_force = np.zeros_like(u)
        dissipation = self.get_dissipation(u=u,dH=dH)
        F = self.External_force(t)
        if F.ndim == 1:
            F = F.reshape(1, 2) 
        external_force[:, 2:] = F  
        u_dot = dH@self.S.T + dissipation +external_force
        return u_dot


    def get_dissipation(self,u,dH=None):
        if dH is None:
            dH = self.Hamiltonian_grad(u.T).T
        dissipation = np.zeros_like(u)
        if dH.ndim ==1:
            dissipation[:,2] = - 0.2* dH[2]
            dissipation[:,3] = -0.07 * dH[3]
        else:
            dissipation[:,2] = - 0.2* dH[:,2]
            dissipation[:,3] = -0.07 * dH[:,3]
        return dissipation


    def sample_trajectory(self,t,u0=None, integrator =  "symplectic midpoint"):
        if u0 is None:
            u0 = self.initial_condition() 

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

  

