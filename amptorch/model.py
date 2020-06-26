"""model.py: Constructs a model consisting of element specific Neural
Networks as understood from Behler and Parrinello's works. A model instance is
constructed based off the unique number of atoms in the dataset."""

import torch
import torch.nn as nn
from torch.nn import Tanh
from torch.autograd import grad
try:
    from torch_sparse import spmm
    spmm_exists = True
except:
    spmm_exists = False
    pass

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"

class MLP(nn.Module):
    """Constructs a fully connected neural network model to be utilized for
    each element type'''

    Arguments:
        n_input_nodes: Number of input nodes (Default=20 using BP SF)
        n_output_nodes: Number of output nodes (Default=1)
        n_layers: Total number of layers in the neural network
        n_hidden_size: Number of neurons within each hidden layer
        activation: Activation function to be utilized. (Default=Tanh())
    """

    def __init__(
        self,
        n_input_nodes,
        n_layers,
        n_hidden_size,
        activation,
        n_output_nodes=1,
    ):
        super(MLP, self).__init__()
        if isinstance(n_hidden_size, int):
            n_hidden_size = [n_hidden_size] * (n_layers)
        self.n_neurons = [n_input_nodes] + n_hidden_size + [n_output_nodes]
        self.activation = activation
        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(self.n_neurons[_], self.n_neurons[_ + 1]))
            layers.append(activation())
        layers.append(nn.Linear(self.n_neurons[-2], self.n_neurons[-1]))
        self.model_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """Feeds data forward in the neural network

        Arguments:
            inputs (torch.Tensor): NN inputs
        """
        return self.model_net(inputs)


class BPNN(nn.Module):
    """Combines element specific NNs into a model to predict energy of a given
    structure

    """

    def __init__(
        self, unique_atoms, architecture, device, forcetraining,
        activation=Tanh, require_grd=True
    ):
        super(BPNN, self).__init__()
        self.device = device
        self.req_grad = require_grd
        self.forcetraining = forcetraining
        self.architecture = architecture
        self.activation_fn = activation

        input_length = architecture[0]
        n_layers = architecture[1]
        n_hidden_size = architecture[2]
        self.elementwise_models = nn.ModuleDict()
        for element in unique_atoms:
            self.elementwise_models[element] = MLP(
                n_input_nodes=input_length,
                n_layers=n_layers,
                n_hidden_size=n_hidden_size,
                activation=activation,
            )

    def forward(self, inputs):
        """Forward pass through the model - predicting energy and forces
        accordingly.

        N - Number of training images
        Q - Atoms in batch
        P - Length of fingerprint"""

        with torch.enable_grad():
            
            input_data, batch_size, batch_elements, fprimes, rearange, num_atoms = inputs
            

            if self.device == 'cpu':
                energy_pred = torch.zeros(batch_size, 1).to(self.device)
            else:
                energy_pred = torch.cuda.FloatTensor(batch_size, 1)
            force_pred = torch.tensor([])
            # Constructs an Nx1 empty tensor to store element energy contributions
            if self.forcetraining:
                dE_dFP = torch.tensor([]).to(self.device)
                idx = torch.tensor([]).to(self.device)
            for index, element in enumerate(batch_elements):
                model_inputs = input_data[element][0]
                model_inputs.requires_grad = True
                contribution_index = torch.tensor(input_data[element][1]).to(self.device)
                atomwise_outputs = self.elementwise_models[element].forward(model_inputs)
#                 print(model_inputs)
#                 print(atomwise_outputs)
#                 print(energy_pred)
                
                energy_pred.index_add_(0, contribution_index, atomwise_outputs)
        
#                 energy_pred = torch.div(energy_pred, num_atoms)
                
                
                if self.forcetraining:
                    gradients = grad(
                        energy_pred,
                        model_inputs,
                        grad_outputs=torch.ones_like(energy_pred),
                        create_graph=True,
                    )[0]
                    dE_dFP = torch.cat((dE_dFP, gradients))
                    idx = torch.cat((idx, contribution_index.float()))
            if self.forcetraining:
                """Constructs a 1xPQ tensor that contains the derivatives with respect to
                each atom's fingerprint"""
                boolean = idx[:, None] == torch.unique(idx)
                ordered_idx = torch.nonzero(boolean.t(), as_tuple=False)[:, -1]
                dE_dFP = torch.index_select(dE_dFP, 0, ordered_idx)
                dE_dFP = torch.index_select(dE_dFP, 0, rearange).reshape(1, -1)
                """Sparse multiplication requires the first matrix to be
                sparse.
                Multiplies a 3QxPQ tensor with a PQx1 tensor to return a 3Qx1 tensor
                containing the x,y,z directional forces for each atom"""
                if spmm_exists:
                    dE_dFP = dE_dFP.t()
                    fprimes = fprimes.t()
                    dim1, dim2 = fprimes.shape
                    fprime_t_idx = fprimes._indices()
                    fprime_t_val = fprimes._values()
                    force_pred = -1 * spmm(fprime_t_idx, fprime_t_val, dim1, dim2, dE_dFP)
                else:
                    force_pred = -1 * torch.sparse.mm(fprimes.t(), dE_dFP.t())
                """Reshapes the force tensor into a Qx3 matrix containing all the force
                predictions in the same order and shape as the target forces calculated
                from AMP."""
                force_pred = force_pred.reshape(-1, 3)
        return energy_pred, force_pred

class CustomMSELoss(nn.Module):
    """Custom loss function to be optimized by the regression. Includes aotmic
    energy and force contributions.

    Eq. (26) in A. Khorshidi, A.A. Peterson /
    Computer Physics Communications 207 (2016) 310-324"""

    def __init__(self, force_coefficient=0):
        super(CustomMSELoss, self).__init__()
        self.alpha = force_coefficient

    def forward(
        self,
        prediction,
        target):

        energy_pred = prediction[0]
        energy_targets_per_atom = target[0]
        num_atoms = target[1]
#         energy_pred_per_atom = torch.div(energy_pred, num_atoms)
        
        MSE_loss = nn.MSELoss(reduction="mean")
        energy_loss = MSE_loss(energy_pred, energy_targets_per_atom)

        if self.alpha > 0:
            force_pred = prediction[1]
            if force_pred.nelement() == 0:
                raise Exception('Force training disabled. Set force_coefficient to 0')
            force_targets_per_atom = target[-1]
            num_atoms_extended = torch.cat([idx.repeat(int(idx)) for idx in num_atoms])
            num_atoms_extended = torch.sqrt(
                num_atoms_extended.reshape(-1, 1)
            )
            force_pred_per_atom = torch.div(force_pred, num_atoms_extended)
            force_targets_per_atom = force_targets_per_atom*num_atoms_extended
            force_loss = (self.alpha / 3) * MSE_loss(
                force_pred_per_atom, force_targets_per_atom
            )
            loss = 0.5 * (energy_loss + force_loss)
        else:
            loss = 0.5 * energy_loss
        return loss

class MAELoss(nn.Module):
    def __init__(self, force_coefficient=0):
        super(MAELoss, self).__init__()
        self.alpha = force_coefficient

    def forward(
        self,
        prediction,
        target):

        energy_pred = prediction[0]
        energy_targets = target[0]
        num_atoms = target[1]
        MAE_loss = nn.L1Loss(reduction="sum")
#         energy_per_atom = torch.div(energy_pred, num_atoms)
        #targets_per_atom = torch.div(energy_targets, num_atoms)
        energy_loss = MAE_loss(energy_pred, targets_per_atom)

        if self.alpha > 0:
            force_pred = prediction[1]
            if force_pred.nelement() == 0:
                raise Exception('Force training disabled. Set force_coefficient to 0')
            force_targets = target[-1]
            num_atoms_force = torch.cat([idx.repeat(int(idx)) for idx in num_atoms])
            num_atoms_force = num_atoms_force.reshape(len(num_atoms_force), 1)
            force_pred_per_atom = torch.div(force_pred, num_atoms_force)
            force_targets_per_atom = torch.div(force_targets, num_atoms_force)
            force_loss = (self.alpha / 3) * MAE_loss(
                force_pred_per_atom, force_targets_per_atom
            )
            loss = 0.5 * (energy_loss + force_loss)
        else:
            loss = 0.5 * energy_loss
        return loss

class HuberLoss(nn.Module):
    def __init__(self, force_coefficient=0):
        super(HuberLoss, self).__init__()
        self.alpha = force_coefficient

    def forward(
        self,
        prediction,
        target):

        energy_pred = prediction[0]
        energy_targets = target[0]
        num_atoms = target[1]
        huber_loss = nn.SmoothL1Loss(reduction="none")
        energy_per_atom = torch.div(energy_pred, num_atoms)
        targets_per_atom = torch.div(energy_targets, num_atoms)
        energy_loss = huber_loss(energy_per_atom, targets_per_atom)
        energy_loss = torch.sum(energy_loss)

        if self.alpha > 0:
            force_pred = prediction[1]
            if force_pred.nelement() == 0:
                raise Exception('Force training disabled. Set force_coefficient to 0')
            force_targets = target[-1]
            num_atoms_force = torch.cat([idx.repeat(int(idx)) for idx in num_atoms]).reshape(-1, 1)
            force_pred_per_atom = torch.div(force_pred, num_atoms_force)
            force_targets_per_atom = torch.div(force_targets, num_atoms_force)
            force_loss = (self.alpha / 3) * huber_loss(
                force_pred_per_atom, force_targets_per_atom
            )
            force_loss = torch.sum(force_loss)
            loss = energy_loss + force_loss
        else:
            loss = energy_loss
        return loss
