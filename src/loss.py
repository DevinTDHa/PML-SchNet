import torch


def energy_force_loss(E_pred, E, R, F):
    rho = 0.01
    dist_E = rho * (E - E_pred).abs() ** 2

    dEdR = -torch.autograd.grad(
        E_pred, R, grad_outputs=torch.ones_like(E_pred), retain_graph=True
    )[0]
    dEdR.requires_grad_()

    dist_F = (torch.norm((F.view(-1, 3) - dEdR.view(-1, 3)), dim=1) ** 2).mean()
    return (dist_E + dist_F).mean()
