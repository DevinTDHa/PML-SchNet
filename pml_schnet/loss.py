import torch


def derive_force(E_pred, R):
    return -torch.autograd.grad(
        E_pred,
        R,
        grad_outputs=torch.ones_like(E_pred),
        retain_graph=True,
        create_graph=True,
    )[0]


def energy_force_loss(E_pred, R, E, F, return_force_labels=False):
    rho = 0.01
    dist_E = rho * (E - E_pred) ** 2
    batch_size = len(E_pred)

    dEdR = derive_force(E_pred, R)
    dEdR.requires_grad_()

    diff_F = F.view(batch_size, -1, 3) - dEdR.view(batch_size, -1, 3)
    dist_F = torch.norm(diff_F, dim=-1)
    dist_F_mean = (dist_F**2).mean(axis=1)
    mean_total_dist = (dist_E + dist_F_mean).mean()
    if return_force_labels:
        return mean_total_dist, dEdR
    else:
        return mean_total_dist
