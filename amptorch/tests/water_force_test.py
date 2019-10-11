"""
Exact Gaussian-neural scheme forces and energies of five different non-periodic
configurations and three different periodic configurations have been calculated
in Mathematica, and are given below.  This script checks the values calculated
by the code with and without fortran modules.


FullNN weights must be initialized to 0.5
Output Layer must be Tanh instead of Linear
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ase
from ase import Atoms
from ase.calculators.emt import EMT
from collections import OrderedDict
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.utilities import hash_images
from amp.model import calculate_fingerprints_range
from amp_pytorch import core
from amp_pytorch.mdata import AtomsDataset, factorize_data, collate_amp
from amp_pytorch.NN_model import FullNN


if not os.path.exists('results'):
    os.mkdir('results')

def test_data():
    """Gaussian/Neural non-periodic standard.

    Checks that the answer matches that expected from previous Mathematica
    calculations.
    """

    # Making the list of non-periodic images
    IMAGES = ase.io.read("water.extxyz", ":")
    images = []
    for i in IMAGES[:400]:
        images.append(i)

    # Parameters
    Gs = {
        "O": [
            {"type": "G2", "element": "Pd", "eta": 0.8},
            {
                "type": "G4",
                "elements": ["Pd", "Pd"],
                "eta": 0.2,
                "gamma": 0.3,
                "zeta": 1,
            },
            {
                "type": "G4",
                "elements": ["O", "Pd"],
                "eta": 0.3,
                "gamma": 0.6,
                "zeta": 0.5,
            },
        ],
        "H": [
            {"type": "G2", "element": "Pd", "eta": 0.2},
            {
                "type": "G4",
                "elements": ["Pd", "Pd"],
                "eta": 0.9,
                "gamma": 0.75,
                "zeta": 1.5,
            },
            {
                "type": "G4",
                "elements": ["O", "Pd"],
                "eta": 0.4,
                "gamma": 0.3,
                "zeta": 4,
            },
        ],
    }

    hiddenlayers = {"O": (2,), "H": (2,)}

    hashed_images = hash_images(images)
    descriptor = Gaussian(cutoff=6.5, Gs=Gs, fortran=False)
    descriptor.calculate_fingerprints(hashed_images, calculate_derivatives=True)
    fingerprints_range = calculate_fingerprints_range(descriptor, hashed_images)

    device = "cpu"
    dataset = AtomsDataset(images, descriptor, forcetraining=True, cores=1, lj_data=None)
    fp_length = dataset.fp_length()

    weights = OrderedDict(
        [
            (
                "O",
                OrderedDict(
                    [
                        (1, np.array([[0.5, 0.5]] * (fp_length + 1))),
                        (2, np.array([[0.5], [0.5], [0.5]])),
                    ]
                ),
            ),
            (
                "H",
                OrderedDict(
                    [
                        (1, np.array([[0.5, 0.5]] * (fp_length + 1))),
                        (2, np.array([[0.5], [0.5], [0.5]])),
                    ]
                ),
            ),
        ]
    )

    scalings = OrderedDict(
        [
            ("O", OrderedDict([("intercept", 0), ("slope", 1)])),
            ("H", OrderedDict([("intercept", 0), ("slope", 1)])),
        ]
    )

    # Testing pure-python and fortran versions of Gaussian-neural force call
    device = "cpu"
    dataset = AtomsDataset(images, descriptor, forcetraining=True, cores=1,
            lj_data=None)
    fp_length = dataset.fp_length()
    unique_atoms = dataset.unique()

    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size, collate_fn=collate_amp, shuffle=False)
    model = FullNN(unique_atoms, [fp_length, 2, 2], device, forcetraining=True)
    for batch in dataloader:
        input_data = [batch[0], len(batch[1])]
        for element in unique_atoms:
            input_data[0][element][0] = (
                input_data[0][element][0].to(device).requires_grad_(True)
            )
        fp_primes = batch[3]
        energy_pred, force_pred = model(input_data, fp_primes)
        # print(energy_pred)
        # print(force_pred)

    calc = Amp(
        descriptor,
        model=NeuralNetwork(
            hiddenlayers=hiddenlayers,
            weights=weights,
            scalings=scalings,
            activation="tanh",
            fprange=fingerprints_range,
            mode="atom-centered",
            fortran=False,
        ),
    )

    amp_energies = [calc.get_potential_energy(image) for image in images]
    amp_forces = [calc.get_forces(image) for image in images]
    amp_forces = np.concatenate(amp_forces)
    # print(amp_energies)
    # print(force_pred)

    for idx, i in enumerate(amp_energies):
        assert round(i, 4) == round(
            energy_pred.tolist()[idx][0], 4
        ), "The predicted energy of image %i is wrong!" % (idx + 1)
    for idx, sample in enumerate(amp_forces):
        for idx_d, value in enumerate(sample):
            prediction = force_pred.tolist()[idx][idx_d]
            assert abs(value - prediction) <= 0.001 , (
                "The predicted force of image % i, direction %i is wrong: %s vs %s!"
                % (idx + 1, idx_d, value, force_pred.tolist()[idx][idx_d])
            )
    print("All tests passed!")


if __name__ == "__main__":
    test_data()