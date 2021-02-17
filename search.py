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
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from collections import Counter
import time
from random import sample


def attack_until(fmodel, image, label, attacks):
    # print("\n\n")

    # print(image, type(image))
    # print(label, type(label))
    # print("\n\n")
    image = ep.astensors(
        torch.as_tensor([image.raw.numpy()]))[0]
    label = ep.astensors(torch.as_tensor([np.asscalar(label.raw.numpy())]))[0]
    pred = label

    while pred == label.raw.numpy()[0]:
        attack_1 = sample(attacks, 1)[0]
        raw_advs, _, _ = attack_1(
            fmodel, image, label, epsilons=epsilons)
        image = raw_advs[0]
        ori_predictions = fmodel(image).argmax(axis=-1)
        res_predictions = ori_predictions[0]
        ori_predictions = ori_predictions.raw.numpy()
        pred = ori_predictions[0]
    return res_predictions, image


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
    if dataset == 'mnist':
        model = LeNet()
        model.load_state_dict(torch.load('LeNet'))
        model.eval()
    else:

        model = torch.hub.load('pytorch/vision:v0.6.0',
                               'resnet50', pretrained=True)
        model.eval()

    fmodel = PyTorchModel(model, bounds=(0, 1))

    images, labels = ep.astensors(
        *samples(fmodel, dataset=dataset, batchsize=batch))

    attacks_1 = [
        fa.L2FastGradientAttack(),
        fa.L2DeepFoolAttack(),
        # fa.L2CarliniWagnerAttack(),
        fa.DDNAttack(),

        fa.LinfBasicIterativeAttack(),
        fa.LinfFastGradientAttack(),
        fa.LinfDeepFoolAttack(),
        fa.LinfPGD(),
    ]
    attacks_2 = [
        fa.L2FastGradientAttack(),
        fa.L2DeepFoolAttack(),
        # fa.L2CarliniWagnerAttack(),
        fa.DDNAttack(),

        fa.LinfBasicIterativeAttack(),
        fa.LinfFastGradientAttack(),
        fa.LinfDeepFoolAttack(),
        fa.LinfPGD(),
    ]

    attacks_result_total, labels_total, attacks_result, drop_result = [0 for _ in range(len(attacks_1))], [
        0 for _ in range(len(attacks_1))], [0 for _ in range(len(attacks_1))], [0 for _ in range(len(attacks_1))]

    for n, attack_1 in enumerate(attacks_1):
        print(n, str(attack_1)[:8])
        recovered = 0
        # which needs to be the same as label
        ori_predictions = fmodel(images).argmax(axis=-1)
        raw_advs, _, _ = attack_1(
            fmodel, images, labels, epsilons=epsilons)
        # print("\n\n")
        # print(type(images), images)
        # print(type(labels), labels)
        # print("\n\n")
        raw_advs = raw_advs[0]
        adv_predictions = fmodel(raw_advs).argmax(axis=-1)
        drop_list = []
        # only filter adversarial ones
        for i in range(batch):
            if ori_predictions[i].raw.numpy() == adv_predictions[i].raw.numpy():
                drop_list.append(i)
        if batch == len(drop_list):
            continue
        # for each image
        images_double_adv_predictions = []
        images_double_advs = []
        final_predictions = []
        # created adversarial image for each image with an attack
        for attack_2 in attacks_2:
            double_advs, _, _ = attack_2(
                fmodel, raw_advs, adv_predictions, epsilons=epsilons)
            double_advs = double_advs[0]

            double_adv_predictions = fmodel(
                double_advs).argmax(axis=-1)
            double_adv_predictions_ = fmodel(double_advs)
            images_double_adv_predictions.append(
                (double_adv_predictions, double_adv_predictions_))
            images_double_advs.append(double_advs)
        # for each image
        for i in range(batch):
            if i in drop_list:
                final_predictions.append([10000])
                continue
            adv_pred = adv_predictions[i].raw.numpy()
            image_predictions = [a[0][i]
                                 for a in images_double_adv_predictions]
            image_predictions_rest = [a[1][i]
                                      for a in images_double_adv_predictions]
            image_ = [a[i] for a in images_double_advs]
            for j in range(len(attacks_2)):
                # if recovery did not change the label yet
                if image_predictions[j].raw.numpy() == adv_pred:
                    image_predictions[j], image_[
                        j] = attack_until(fmodel, image_[j], image_predictions[j], attacks_2)
            image_predictions = [np.asscalar(
                a.raw.numpy()) for a in image_predictions]
            final_predictions.append(
                (image_predictions, image_predictions_rest))

        for j in range(batch):
            if j in droplist:
                continue
            labels_.append(labels[j])
            final_predictions_.append(final_predictions[j])

        labels_total[n] = labels_
        attacks_result_total[n] = final_predictions_

        final_predictions = [Counter(a).most_common()[0][0]
                             for a in final_predictions]
        for i in range(batch):
            if i in drop_list:
                continue
            # print(final_predictions[i], np.asscalar(ori_predictions[i].raw.numpy(
            # )), final_predictions[i] == np.asscalar(ori_predictions[i].raw.numpy()))
            if final_predictions[i] == np.asscalar(ori_predictions[i].raw.numpy()):
                recovered += 1

        attacks_result[n] = recovered
        drop_result[n] = batch-len(drop_list)
    print("-------------------------------------------------------")
    print("Dataset(epsilon = {0}): ".format(epsilons[0]), dataset)
    print("recovery matrix: ", attacks_result)
    print(" total  matrix:  ", drop_result)
    print("recovery rate: ", sum(attacks_result)/sum(drop_result))
    print("    labels: ", labels_total)
    print("attacks_result_total: ", attacks_result_total)
    print("-------------------------------------------------------")


if __name__ == "__main__":
    # datasets = ['imagenet', 'cifar10', 'cifar100']
    datasets = ['mnist']
    batch = 20
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

        0.8,

        # 1.0,
    ]
    for dataset in datasets:
        for epsilon in epsilons:
            run(dataset, batch, [epsilon])
