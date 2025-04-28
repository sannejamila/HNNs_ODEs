import numpy as np
import torch
from NumericalIntegration.Numerical_Integration import *
from tqdm import tqdm
import matplotlib.pyplot as plt
"""

def generate_data(ntrajectories, t_sample,system,integrator, true_derivatives = False,H0=None,u0s=None,data_type = torch.float32):
    #Parameters
    nstates = system.nstates
    traj_length = t_sample.shape[0] 

    #Initializing 
    u = np.zeros((ntrajectories,traj_length,nstates))
    dudt = np.zeros_like(u)
    t = np.zeros((ntrajectories,traj_length))

    u0_ = np.zeros((ntrajectories,nstates))

    for i in tqdm(range(ntrajectories)):
        if u0s is not None:
            u0 = np.array(u0s[i])
            u[i], dudt[i], t[i],u0_[i] = system.sample_trajectory(t=t_sample,u0=u0,integrator=integrator)
        else:
            u[i], dudt[i], t[i],u0_[i] = system.sample_trajectory(t=t_sample,integrator=integrator)
    
    #Reshaping
    dt = torch.tensor([t[0, 1] - t[0, 0]], dtype=data_type)
    u_start = torch.tensor(u[:, :-1], dtype=data_type).reshape(-1, nstates)
    u_end = torch.tensor(u[:, 1:], dtype=data_type).reshape(-1, nstates)
    t_start = torch.tensor(t[:, :-1], dtype=data_type).reshape(-1, 1)
    dt = dt * torch.ones_like(t_start, dtype=data_type)

    if true_derivatives:
        dudt = torch.tensor(dudt[:, :-1], dtype=data_type).reshape(-1, 1, nstates)
    else:
        dudt = (u_end - u_start).clone().detach() / dt[0, 0]

    u_exact = u
    return (u_start, u_end, dt), dudt, u_exact, u0_
"""
def generate_data(system, ntrajectories, t_sample, integrator, true_derivatives=False, u0s=None, data_type=torch.float32):
    nstates = system.nstates
    traj_length = t_sample.shape[0]

    u = np.zeros((ntrajectories,traj_length,nstates))
    dudt = np.zeros_like(u)
    t = np.zeros((ntrajectories,traj_length))
    u0_ = np.zeros((ntrajectories, nstates))

    for i in tqdm(range(ntrajectories)):
        if u0s is not None:
            u0 = np.array(u0s[i])
            u[i], dudt[i], t[i],u0_[i] = system.sample_trajectory(t=t_sample,u0=u0,integrator=integrator)
        else:
            u[i], dudt[i], t[i],u0_[i] = system.sample_trajectory(t=t_sample,integrator=integrator)

    dt = torch.tensor([t[0, 1] - t[0, 0]], dtype=data_type)
    u_start = torch.tensor(u[:, :-1], dtype=data_type).reshape(-1, nstates)
    u_end = torch.tensor(u[:, 1:], dtype=data_type).reshape(-1, nstates)
    t_start = torch.tensor(t[:, :-1], dtype=data_type).reshape(-1, 1)
    dt = dt * torch.ones_like(t_start)

    dudt_tensor = torch.tensor(dudt[:, :-1], dtype=data_type).reshape(-1, nstates) if true_derivatives else (u_end - u_start) / dt[0, 0]

    return (u_start, u_end, t_start, dt), dudt_tensor, u, u0_


def find_crossings(u):
    x = u[:, 0]
    crossings = []
    
    for i in range(len(x) - 1):
        if x[i] * x[i + 1] < 0:  #Change in sign
            #Interpolation
            alpha = abs(x[i]) / (abs(x[i]) + abs(x[i + 1]))
            index_exact = i + alpha
            crossings.append(index_exact)
    
    return np.array(crossings)

def plot_poincare(u, desc = "Predicted Poincaré Sections"):
    event_indices = np.round(find_crossings(u[0])).astype(int)
    y_py = np.array([u[0][:,2][event_indices], u[0][:,3][event_indices]]).T
    plt.scatter(y_py[0], y_py[1], s=1, label = desc)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.legend()
    plt.xlabel("y")
    plt.ylabel("p_y")
    plt.title("Poincaré Sections")
    plt.grid(True)
