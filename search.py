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
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
import time
from random import sample

image_predictions[j], image_[
    j] = attack_until(image_[j], adv_pred)


def attack_until(fmodel, image, label):
    pred = label
    attacks = [
        fa.L2FastGradientAttack(),
        fa.L2DeepFoolAttack(),
        # fa.L2CarliniWagnerAttack(),
        fa.DDNAttack(),

        # L_inf
        fa.LinfBasicIterativeAttack(),
        fa.LinfFastGradientAttack(),
        fa.LinfDeepFoolAttack(),
        fa.LinfPGD(),
    ]
    while pred == label:
        attack_1 = sample(attacks, 1)
        raw_advs, _, _ = attack_1(
            fmodel, image, label, epsilons=epsilons)
        image = raw_advs[0]
        ori_predictions = fmodel(image).argmax(axis=-1)
        ori_predictions = ori_predictions.raw.numpy()
        pred = ori_predictions[0]
    return pred, image


def run(dataset, batch=20, epsilons=[0.1]):
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

    attacks = [
        fa.L2FastGradientAttack(),
        fa.L2DeepFoolAttack(),
        # fa.L2CarliniWagnerAttack(),
        fa.DDNAttack(),

        fa.LinfBasicIterativeAttack(),
        fa.LinfFastGradientAttack(),
        fa.LinfDeepFoolAttack(),
        fa.LinfPGD(),
    ]
    attacks_result = [[0 for _ in range(len(attacks))]
                      for _ in range(len(attacks))]
    drop_result = [[0 for _ in range(len(attacks))]
                   for _ in range(len(attacks))]
    total_attack_fail_drop, total_predict_fail_drop, total_drop, total_recovered = 0, 0, 0, 0

    for n, attack_1 in enumerate(attacks):
        # which needs to be the same as label
        ori_predictions = fmodel(images).argmax(axis=-1)
        raw_advs, _, _ = attack_1(
            fmodel, images, labels, epsilons=epsilons)

        # print(type(labels), labels)
        raw_advs = raw_advs[0]
        adv_predictions = fmodel(raw_advs).argmax(axis=-1)
        drop_list = []
        # only filter adversarial ones
        for i in range(batch):
            if ori_predictions[i].raw.numpy() != adv_predictions[i].raw.numpy():
                drop_list.append(i)
        if batch == len(drop_list):
            continue
        # for each image
        images_double_adv_predictions = []
        images_double_advs = []

        # created adversarial image for each image with an attack
        for attack_2 in attacks:
            double_advs, _, _ = attack_2(
                fmodel, raw_advs, adv_predictions, epsilons=epsilons)
            double_advs = double_advs[0]
            double_adv_predictions = fmodel(
                double_advs).argmax(axis=-1)
            images_double_adv_predictions.append(double_adv_predictions)
            images_double_advs.append(double_advs)
            # double_adv_predictions = double_adv_predictions.raw.numpy()
        # for each image
        for i in range(batch):
            if i in drop_list:
                continue
            adv_pred = adv_predictions[i].raw.numpy()
            image_predictions = [a[i] for a in images_double_adv_predictions]
            image_ = [a[i] for a in images_double_advs[i]]

            for j in range(len(attacks)):
                if image_predictions[j].raw.numpy() == adv_pred:
                    image_predictions[j], image_[
                        j] = attack_until(fmodel, image_[j], adv_pred)

        final_predictions = [Counter(a).most_common()[0][1]
                             for a in image_predictions]
        for i in range(batch):
            if i in drop_list:
                drop += 1
                continue
            if final_predictions[i] == ori_predictions[i]:
                recovered += 1

        drop_result[n][m] = batch-drop
        attacks_result[n][m] = recovered
        total_drop += drop
        total_recovered += recovered
    '''     
    print("-------------------------------------------------------")
    print("Dataset(epsilon = {0}): ".format(epsilons[0]), dataset)
    print("recovery matrix: ", attacks_result)
    print(" total  matrix:  ", drop_result)
    print("  Drop report:   ", attack_fail_drop, predict_fail_drop)
    print("recovery rate: ", total_recovered /
          (batch*len(attacks)**2-total_drop))
    print("-------------------------------------------------------")
    '''


if __name__ == "__main__":
    # datasets = ['mnist', 'imagenet', 'cifar10', 'cifar100']
    datasets = ['cifar100']
    batch = 5
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

        # 0.8,

        # 1.0,
    ]
    for dataset in datasets:
        for epsilon in epsilons:
            run(dataset, batch, [epsilon])
