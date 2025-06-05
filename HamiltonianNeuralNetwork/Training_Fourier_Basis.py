import torch
from HamiltonianNeuralNetwork.pHNN import *
from HamiltonianNeuralNetwork.Train import *


def get_optimal_period(F_model, candidates, t_vals):
    errors = []

    for a in candidates:
        t_shifted = t_vals + a.item()
        mask = t_shifted <= t_vals[-1]  

        t_base = t_vals[mask].reshape(-1, 1)
        t_shift = t_shifted[mask].reshape(-1, 1)

        f_base = F_model(t_base).detach()
        f_shift = F_model(t_shift).detach()

        err = ((f_base - f_shift) ** 2).mean().item()
        errors.append(err)

    errors = torch.tensor(errors)
    return candidates[errors.argmin()], errors

def Retrain_with_Forier_Basis(model,candidates, integrator, train_data,t_vals, val_data,sys,lr, batch_size,epochs,loss_func,penalty_func,schedule = True):

    Hamiltonian_est = model.Hamiltonian_est
    External_Forces_est = model.External_Forces_est

   
    External_Forces_est.Fourier = False
    optimal_period = get_optimal_period(External_Forces_est, candidates, t_vals)[0]
    External_Forces_est.learned_period = optimal_period
    External_Forces_est.Fourier = True

    model_fourier = PortHamiltonianNeuralNetwork(
        nstates=4,
        S=sys.S,
        Hamiltonian_est=Hamiltonian_est,
        External_Forces_est=External_Forces_est
    )
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fourier.parameters()), lr=lr)
    optimizer = torch.optim.AdamW(model_fourier.parameters(), lr=lr)
    if schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    trainer = Training(
        model=model_fourier,
        integrator=integrator,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        system=sys,
        batch_size=batch_size,
        epochs=epochs
    )

    model_fourier, training_details = trainer.train(loss_func=loss_func, penalty_func=penalty_func,scheduler=scheduler)

    return model_fourier, training_details