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


def run(dataset, batch=20):
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
    # print(type(images), type(labels))
    # print(images[0], labels[0])
    # clean_acc = accuracy(fmodel, images, labels)
    # print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    attacks = [
        fa.L2FastGradientAttack(),
        fa.L2DeepFoolAttack(),
        fa.L2CarliniWagnerAttack(),
        fa.DDNAttack(),

        # L_inf
        fa.LinfBasicIterativeAttack(),
        fa.LinfFastGradientAttack(),
        fa.LinfDeepFoolAttack(),
        fa.LinfPGD(),

        # fa.BoundaryAttack(),  # reduce perturbation while staying adversarial

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

    attacks_result = [[0 for _ in range(len(attacks))]
                      for _ in range(len(attacks))]
    drop_result = [[0 for _ in range(len(attacks))]
                   for _ in range(len(attacks))]
    total_drop, total_recovered = 0, 0
    for n, attack_1 in enumerate(attacks):
        print(n, str(attack_1)[:8])
        for m, attack_2 in enumerate(attacks):
            # which needs to be the same as label
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

            attack_fail_drop, predict_fail_drop, recovered = 0, 0, 0
            ori_predictions = ori_predictions.raw.numpy()
            adv_predictions = adv_predictions.raw.numpy()
            double_adv_predictions = double_adv_predictions.raw.numpy()

            for i in range(batch):
                # if n == 3 or n == 4:
                # print(str(attack_1)[:8], str(attack_2)[
                #       :8], ori_predictions[i], adv_predictions[i], double_adv_predictions[i])

                #  or ori_predictions[i] != labels[i]
                if ori_predictions[i] == adv_predictions[i]:
                    attack_fail_drop += 1
                    continue  # attack failed at all
                elif ori_predictions[i] != labels[i].raw.numpy():
                    predict_fail_drop += 1
                    continue
                else:
                    if ori_predictions[i] == double_adv_predictions[i]:
                        recovered += 1
            drop = attack_fail_drop + predict_fail_drop
            drop_result[n][m] = batch-drop
            attacks_result[n][m] = recovered
            total_drop += drop
            total_recovered += recovered
    print("-------------------------------------------------------")
    print("Dataset: ", dataset)
    print("recovery matrix: ", attacks_result)
    print(" total  matrix:  ", drop_result)
    print("  Drop report:   ", attack_fail_drop, predict_fail_drop)
    print("recovery rate: ", total_recovered /
          (batch*len(attacks)**2-total_drop))
    print("-------------------------------------------------------")


if __name__ == "__main__":
    datasets = ['mnist', 'imagenet', 'cifar10', 'cifar100']
    for dataset in datasets:
        run(dataset, 20)
