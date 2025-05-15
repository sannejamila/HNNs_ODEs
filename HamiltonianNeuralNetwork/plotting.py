import torch
import numpy as np
import matplotlib.pyplot as plt

def contour_plot_Hamiltonian(model,sys,u_pred):
    u0 = u_pred[0]
    C = model.Hamiltonian(torch.tensor(u0, dtype = torch.float32))- sys.Hamiltonian(u0)

    x = u_pred[:, 0]
    y = u_pred[:, 1]
    px = np.zeros_like(x)  
    py = np.zeros_like(y) 

    X, Y = np.meshgrid(x, y)

    x_flattened = X.flatten()
    y_flattened = Y.flatten()
    px = np.full_like(x_flattened, 0)
    py = np.full_like(y_flattened, 0)
   
    u_samples = np.stack([x_flattened, y_flattened, px, py], axis=1)

    H_true = np.array([sys.Hamiltonian(u) for u in u_samples])
    with torch.no_grad():
       H_learned = torch.tensor([(model.Hamiltonian(torch.tensor(u, dtype = torch.float32)) - C) for u in u_samples])

    #Reshape to grid
    H_true_grid = H_true.reshape(X.shape)
    H_learned_grid = H_learned.reshape(X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    cs1 = axes[0].contour(X, Y, H_true_grid, levels=20, cmap='viridis')
    axes[0].set_title('True Hamiltonian $H(x, y, 0, 0)$')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal', 'box')
    fig.colorbar(cs1, ax=axes[0])

    cs2 = axes[1].contour(X, Y, H_learned_grid, levels=20, cmap='viridis')
    axes[1].set_title('Learned Hamiltonian $\hat{H}_\\theta(x, y, 0, 0)$')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal', 'box')
    fig.colorbar(cs2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_PHNN_prediction(model,u_pred,u_exact,t_sample,sys,u0, integrator):
    H_exact = torch.tensor([sys.Hamiltonian(u) for u in u_pred])
    C = model.Hamiltonian(torch.tensor(u0, dtype = torch.float32))- sys.Hamiltonian(u0.squeeze(0))

    #Computed estimated Hamiltonian from values for prediction
    H_nn= torch.tensor([(model.Hamiltonian(torch.tensor(u))- C)for u in u_pred])

    #Computed true Hamiltonian values for prediction
    H_exact_pred = torch.tensor([sys.Hamiltonian(u) for u in u_pred])
        
    #Computed NN Hamiltonian for exact
    H_nn_exact = torch.tensor([(model.Hamiltonian(torch.tensor(u, dtype = torch.float32)) - C) for u in u_exact.squeeze(0)])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))  
    y, py = u_exact[0][:, 1], u_exact[0][:, 3]
    ax[0].plot(y, py, label="Exact")
    y, py = u_pred[:, 1], u_pred[:, 3]
    ax[0].plot(y, py, label="HNN")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel(r"$p_y$")
    ax[0].set_title(f"Phase Space Trajectory: {integrator}")
    ax[0].legend()

    t = t_sample.squeeze(0)

    ax[1].plot(t, H_exact, label="H(u)")
    ax[1].plot(t, H_nn, label=r"$\hat{H}_{\theta}(\hat u)$")
    ax[1].plot(t, H_exact_pred, label=r"$H(\hat u)$")
    ax[1].plot(t, H_nn_exact, label=r"$\hat{H}_{\theta}(u)$")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("H")
    ax[1].set_title(f"Hamiltonian Comparison: {integrator}")
    ax[1].legend()

    ax[2].plot(t, H_nn, label=r"$\hat{H}_{\theta}(\hat u)$")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("H")
    ax[2].set_title(f"Predicted Hamiltonian: {integrator}")
    ax[2].legend()
    plt.tight_layout() 
    plt.show()

    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(-1) 
    external_force = model.External_Force(t_tensor).detach() 

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))  
    y, py = u_exact[0][:, 1], u_exact[0][:, 3]
    ax[0].plot(y, py, label="Exact")
    y, py = u_pred[:, 1], u_pred[:, 3]
    ax[0].plot(y, py, label="HNN")
    ax[0].set_xlabel("y")
    ax[0].set_ylabel(r"$p_y$")
    ax[0].set_title(f"Phase Space Trajectory: {integrator}" )
    ax[0].legend()


    F = sys.External_force(t)


    ax[1].plot(t, external_force[:,0], label=r"$\hat{F}_{\theta}(t)$")
    #ax[1].plot(t, 0.5 * np.cos(2 * np.pi * t), label=r"${F}_{x}(t)$")
    ax[1].plot(t, F[:,0], label=r"${F}(t)$")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("H")
    ax[1].set_title(f"External Force: {integrator}")
    ax[1].legend()



    ax[2].plot(t, external_force[:,1], label=r"$\hat{F}_{\theta}(t)$")
    #ax[2].plot(t, 0.5 * np.cos(2 * np.pi * t), label=r"${F}_{y}(t)$")
    ax[2].plot(t, F[:,1], label=r"${F}(t)$")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("H")
    ax[2].set_title(f"External Force: {integrator}")
    ax[2].legend()
    plt.tight_layout() 
    plt.show()
