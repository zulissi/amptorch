import sys
import skorch
from skorch.utils import to_numpy
import torch
from torch.nn import MSELoss, L1Loss
import numpy as np
from amptorch.data_preprocess import collate_amp


def target_extractor(y):
    return (
        (to_numpy(y[0]), to_numpy(y[1]))
        if len(y) == 2
        else (to_numpy(y[0]), to_numpy(y[1]), to_numpy(y[2]))
    )


def energy_score(net, X, y):
    mse_loss = MSELoss(reduction="mean")
    energy_pred, _ = net.forward(X)
    
    # Get the energies from the dataset (don't use the cache)
    collate_data = collate_amp(X)
    y = collate_data[1]
    num_atoms = collate_data[0][5]

    
    # Get the scalings
    device = energy_pred.device
    if not hasattr(X, "scalings"):
        X = X.dataset
    scale = X.scalings[-1]
    
    # Get the num atoms from y
#     num_atoms = torch.FloatTensor(np.concatenate(y[1::3])).reshape(-1, 1).to(device)

    energy_targets = torch.tensor(np.concatenate(y[0::3])).to(device).reshape(-1, 1)
    energy_targets = scale.denorm(energy_targets)
    
#     energy_pred_per_atom = torch.div(energy_pred, num_atoms)
    energy_pred = scale.denorm(energy_pred)

    energy_loss = mse_loss(energy_pred, energy_targets)
    #energy_loss /= dataset_size
    energy_rmse = torch.sqrt(energy_loss)
    return energy_rmse


def forces_score(net, X, y):
    mse_loss = MSELoss(reduction='none')
    _, force_pred = net.forward(X)
    if force_pred.nelement() == 0:
        raise Exception("Force training disabled. Disable force scoring!")
    device = force_pred.device
    if not hasattr(X, "scalings"):
        X = X.dataset
    scale = X.scalings[-1]
    num_atoms = torch.FloatTensor(np.concatenate(y[1::3])).reshape(-1, 1).to(device)
    force_targets_per_atom = torch.tensor(np.concatenate(y[2::3])).to(device)
    force_targets_per_atom = scale.denorm(force_targets_per_atom)
    device = force_pred.device
    dataset_size = len(num_atoms)
    num_atoms_extended = torch.cat([idx.repeat(int(idx)) for idx in num_atoms]).reshape(-1, 1)
    force_pred_per_atom = scale.denorm(torch.div(force_pred, num_atoms_extended))
    force_targets = force_targets_per_atom*num_atoms_extended
    force_pred = force_pred_per_atom*num_atoms_extended
    force_mse = mse_loss(force_pred, force_targets)
    force_mse /= 3 * dataset_size * num_atoms_extended
    force_rmse = torch.sqrt(force_mse.sum())
    return force_rmse

class train_end_load_best_loss(skorch.callbacks.base.Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_train_end(self, net, X, y):
        net.load_params("./results/checkpoints/{}_params.pt".format(self.filename))

def make_force_header(log):
    header = "%5s %12s %12s %12s %7s"
    log(header % ("Epoch", "EnergyRMSE", "ForceRMSE", "TrainLoss", "Dur"))
    log(header % ("=" * 5, "=" * 12, "=" * 12, "=" * 12, "=" * 7))


def make_energy_header(log):
    header = "%5s %12s %12s %7s"
    log(header % ("Epoch", "EnergyRMSE", "TrainLoss", "Dur"))
    log(header % ("=" * 5, "=" * 12, "=" * 12, "=" * 7))


def make_val_force_header(log):
    header = "%5s %12s %12s %12s %12s %7s"
    log(header % ("Epoch", "EnergyRMSE", "ForceRMSE", "TrainLoss", "ValidLoss", "Dur"))
    log(header % ("=" * 5, "=" * 12, "=" * 12, "=" * 12, "=" * 12, "=" * 7))


def make_val_energy_header(log):
    header = "%5s %12s %12s %12s %7s"
    log(header % ("Epoch", "EnergyRMSE", "TrainLoss", "ValidLoss", "Dur"))
    log(header % ("=" * 5, "=" * 12, "=" * 12, "=" * 12, "=" * 7))


def log_results(model, log):
    log("Training initiated...")
    if model.train_split != 0:
        if model.criterion__force_coefficient != 0:
            make_val_force_header(log)
            for epoch, ermse, frmse, tloss, vloss, dur in model.history[
                :,
                (
                    "epoch",
                    "energy_score",
                    "forces_score",
                    "train_loss",
                    "valid_loss",
                    "dur",
                ),
            ]:
                log(
                    "%5i %12.4f %12.4f %12.4f %12.4f %7.4f"
                    % (epoch, ermse, frmse, tloss, vloss, dur)
                )
        else:
            make_val_energy_header(log)
            for epoch, ermse, tloss, vloss, dur in model.history[
                :, ("epoch", "energy_score", "train_loss", "valid_loss", "dur")
            ]:
                log(
                    "%5i %12.4f %12.4f %12.4f %7.4f" % (epoch, ermse, tloss, vloss, dur)
                )
    else:
        if model.criterion__force_coefficient != 0:
            make_force_header(log)
            for epoch, ermse, frmse, tloss, dur in model.history[
                :, ("epoch", "energy_score", "forces_score", "train_loss", "dur")
            ]:
                log(
                    "%5i %12.4f %12.4f %12.4f %7.4f" % (epoch, ermse, frmse, tloss, dur)
                )
        else:
            make_energy_header(log)
            for epoch, ermse, tloss, dur in model.history[
                :, ("epoch", "energy_score", "train_loss", "dur")
            ]:
                log("%5i %12.4f %12.4f %7.4f" % (epoch, ermse, tloss, dur))
    log("...Training Complete!\n")


