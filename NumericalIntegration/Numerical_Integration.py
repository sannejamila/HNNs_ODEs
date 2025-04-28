import numpy as np
import torch
from scipy.optimize import newton


def newton_torch(func, guess, threshold=1e-7, max_iters=100, damping=1.0):
    guess = torch.tensor(guess, dtype=torch.float32, requires_grad=True)
    for i in range(max_iters):
        value = func(guess) 
        if torch.linalg.norm(value) < threshold:
            return guess
        J = torch.autograd.functional.jacobian(func, guess)  
        try:
            step = torch.linalg.solve(J, -value)  
        except RuntimeError:
            print("Jacobian is singular, stopping.")
            return guess
        guess = guess + damping * step 
    return guess

def RK4_time_derivative(u_dot,u_start, dt):
    k1 = u_dot(u_start)
    k2 = u_dot(u_start + dt/2*k1)  
    k3 = u_dot(u_start + dt/2*k2)  
    k4 = u_dot(u_start + dt*k3)  
    return 1/6*(k1+2*k2+2*k3+k4)

"""
def symplectic_midpoint_time_derivative(u_dot, u_start, dt, u_end=None, *args, **kwargs):
    if u_end is None:
        def g(u):
            return u - u_start - dt * u_dot(0.5 * (u + u_start), *args, **kwargs)

        if isinstance(u_start, torch.Tensor):
            original_shape = u_start.shape
            u_start = u_start.squeeze(0)
            u_end = newton_torch(g, u_start)
            u_mid = 0.5 * (u_start + u_end)
            return u_dot(u_mid.view(original_shape), *args, **kwargs)
        else:
            u_end = newton(g, u_start)
    u_mid = 0.5 * (u_start + u_end)
    return u_dot(u_mid, *args, **kwargs)    
def explicit_midpoint_time_derivative(u_dot,u_start,dt):
    u_temp = u_start + dt/2*u_dot(u_start)
    lhs = u_dot(u_temp)
    return lhs

def symplectic_midpoint_time_derivative(u_dot,u_start,dt,u_end = None):
    if u_end == None:
        def g(u):
            return u-u_start-dt*u_dot(0.5*(u+u_start))
        if isinstance(u_start,torch.Tensor):
            original_shape = u_start.shape
            u_start = u_start.squeeze(0) 
            u_end = newton_torch(g,u_start)
            u_mid = 0.5*(u_start + u_end)
            return u_dot(u_mid.view(original_shape))
        else:
            u_end = newton(g,u_start, tol = 1e-7, maxiter = 1000)
    u_mid = 0.5*(u_start + u_end)
    lhs = u_dot(u_mid)
    return lhs

def explicit_midpoint_time_derivative_withtime(u_dot,u_start,t_start,dt):
    u_temp = u_start + dt/2*u_dot(u_start,t_start)
    lhs = u_dot(u_temp, t_start+dt/2)
    return lhs

def symplectic_midpoint_time_derivative_withtime(u_dot,u_start, t_start, dt,u_end = None):
    t_end = t_start + dt/2
    if u_end == None:
        def g(u):
            return u-u_start-dt*u_dot(0.5*(u+u_start),t_end)
        if isinstance(u_start,torch.Tensor):
            original_shape = u_start.shape
            u_start = u_start.squeeze(0) 
            u_end = newton_torch(g,u_start)
            u_mid = 0.5*(u_start + u_end)
            return u_dot(u_mid.view(original_shape), t_end)
        else:
            u_end = newton(g,u_start, tol = 1e-7, maxiter = 1000)
    u_mid = 0.5*(u_start + u_end)
    lhs = u_dot(u_mid, t_end)
    return lhs

"""
def explicit_midpoint_time_derivative(u_dot,u_start,dt, *args, **kwargs):
    u_temp = u_start + dt/2*u_dot(u_start, *args, **kwargs)
    lhs = u_dot(u_temp, *args, **kwargs)
    return lhs



def symplectic_midpoint_time_derivative(u_dot, u_start, dt, u_end=None, *args, **kwargs):

    

    if u_end is None:
        def g(u):
            return u - u_start - dt * u_dot(u = 0.5 * (u + u_start), *args, **kwargs)

        if isinstance(u_start, torch.Tensor):
            original_shape = u_start.shape
            u_start = u_start.squeeze(0)
            u_end = newton_torch(g, u_start)
            u_mid = 0.5 * (u_start + u_end)
            return u_dot(u_mid.view(original_shape), *args, **kwargs)
        else:
            u_end = newton(g, u_start)
    u_mid = 0.5 * (u_start + u_end)
    return u_dot(u_mid, *args, **kwargs)



def symplectic_euler(u_dot,u_start,dt):
    g = u_dot(u_start)
    f = u_dot(u_start+u_dot(u_start)*dt)
    if isinstance(u_start,torch.Tensor):
        if u_start.ndim == 1:
            rhs = torch.cat((f[0:2], g[2:4]))
        else:
            rhs = torch.cat((f[:,0:2],g[:,2:4]),axis = 1)
    else:
        rhs = np.concatenate((f[0:2], g[2:4]))
    return rhs