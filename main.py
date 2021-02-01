import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np
import matplotlib.pyplot as plt
import ast
from keras import layers, models, datasets, backend
import keras
import foolbox
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    batch = 2

    class LeNet(nn.Module):

        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def name(self):
            return "LeNet"

    model = LeNet()
    model.load_state_dict(torch.load('LeNet'))
    model.eval()
    fmodel = PyTorchModel(model, bounds=(0, 1))

    images, labels = ep.astensors(
        *samples(fmodel, dataset="mnist", batchsize=batch))
    # clean_acc = accuracy(fmodel, images, labels)
    # print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    attacks = [
        fa.FGSM(),
        fa.LinfPGD(),
        # fa.LinfBasicIterativeAttack(),
        # fa.LinfAdditiveUniformNoiseAttack(),
        # fa.LinfDeepFoolAttack(),
    ]
    epsilons = [
        # 0.0,
        # 0.0005,
        # 0.001,
        # 0.0015,
        # 0.002,
        # 0.003,
        # 0.005,
        # 0.01,
        # 0.02,
        # 0.03,
        0.1,
        # 0.3,
        # 0.5,
        # 1.0,
    ]

    attacks_result = [[0]*len(attacks)]*len(attacks)
    for n, attack_1 in enumerate(attacks):
        for m, attack_2 in enumerate(attacks):
            ori_predictions = fmodel(images).argmax(axis=-1)

            raw_advs, _, success = attack_1(
                fmodel, images, labels, epsilons=epsilons)
            raw_advs = raw_advs[0]
            adv_predictions = fmodel(raw_advs).argmax(axis=-1)

            raw_double_advs, _, success = attack_2(
                fmodel, raw_advs, adv_predictions, epsilons=epsilons)
            raw_double_advs = raw_double_advs[0]
            double_adv_predictions = fmodel(
                raw_double_advs).argmax(axis=-1)

            attack_fail_indices = [i for i, j in zip(
                ori_predictions, adv_predictions) if i == j]
            for ind in attack_fail_indices:
                # drop out failed attack
                del ori_predictions[ind], adv_predictions[ind], double_adv_predictions[ind]
            recovered_values = [i for i, j in zip(
                ori_predictions, double_adv_predictions) if i == j]
            attacks_result[n][m] = len(recovered_values)
    print("recovery matrix: ", attacks_result)
