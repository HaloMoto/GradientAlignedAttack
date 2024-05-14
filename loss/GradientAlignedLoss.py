import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from scipy.optimize import fsolve
from math import exp

def calculate_weights(outputs, ys, outputs_victim):
    outputs_victim_ = outputs_victim.clone().detach()
    prob_victim_ = F.softmax(outputs_victim_, dim=1)
    target_onehot_bool = (torch.nn.functional.one_hot(ys, 1000).float()).bool()
    prob_victim_[target_onehot_bool] = 1.0-prob_victim_[target_onehot_bool]
    prob_victim_ = prob_victim_.cpu().numpy()
    outputs_ = outputs.clone().detach().cpu().numpy()
    ys_ = ys.clone().detach().cpu().numpy()
    ws = []
    for i in range(len(ys_)):
        y = ys_[i]
        def solve_function(unsolved_value):
            w = [unsolved_value[j] for j in range(1000)]
            equations = []
            sum = 0
            for j in range(1000):
                sum += exp(w[j]*outputs_[i,j])
            for j in range(1000):
                if j == y:
                    equations.append((w[j]*sum-w[j]*exp(w[j]*outputs_[i,j]))/sum)
                else:
                    equations.append(w[j]*exp(w[j]*outputs_[i,j])/sum)
            return equations
        solved = fsolve(solve_function, prob_victim_[i])
        ws.append(solved)
    return torch.tensor(ws).to(outputs.device)

class GradientAlignedLoss(nn.Module):
    def __init__(self):
        super(GradientAlignedLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.loss_none = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, ys, outputs_victim, mode='score', reduction=None, topn=None, is_logits=True):
        if mode == 'score':
            outputs_victim_ = outputs_victim.clone().detach()
            if is_logits:
                prob_victim_ = F.softmax(outputs_victim_, dim=1)
            else:
                prob_victim_ = outputs_victim_ / outputs_victim_.sum(dim=1, keepdim=True)
            target_onehot_bool = (torch.nn.functional.one_hot(ys, 1000).float()).bool()
            prob_victim_[target_onehot_bool] = prob_victim_[target_onehot_bool]-1.0
            w = prob_victim_ / torch.log(torch.tensor([2.0])).to(outputs.device)
            if reduction == 'none':
                cost = (w*outputs).sum(1)
            else:
                cost = (w*outputs).mean()
        elif mode == 'prob2':
            w = calculate_weights(outputs, ys, outputs_victim)
            cost = self.loss(w*outputs, ys)
        elif mode == 'topn1':
            outputs_victim_arr = outputs_victim.detach().cpu().numpy()
            topn_max_values = torch.zeros(outputs.shape[0], topn).to(outputs.device)
            topn_max_labels = torch.zeros(outputs.shape[0], topn).to(outputs.device)
            for j in range(topn):
                # ys外的最高概率类别输出值最大化
                if j == 0:
                    max_index = ys.detach().cpu().numpy()
                    min_value = np.min(outputs_victim_arr, axis=1)
                else:
                    max_index = np.argmax(outputs_victim_arr, axis=1)
                    min_value = np.min(outputs_victim_arr, axis=1)
                for i in range(len(max_index)):
                    outputs_victim_arr[i, max_index[i]] = min_value[i]
                topn_labels = np.argmax(outputs_victim_arr, axis=1)
                topn_labels = torch.tensor(topn_labels).to(outputs.device)
                topn_values = outputs_victim_arr.max(axis=1)
                topn_values = torch.tensor(topn_values).to(outputs.device)
                topn_max_labels[:, j] = topn_labels
                topn_max_values[:, j] = topn_values
            topn_max_labels = topn_max_labels.to(torch.int64)
            loss1 = self.loss(outputs, ys)
            ws = F.softmax(topn_max_values, dim=1)
            loss2 = torch.tensor(0.).to(outputs.device)
            for i in range(topn_max_values.shape[1]):
                topn_label = topn_max_labels[:, i]
                w = ws[:, i]
                loss2 += torch.mean(w*self.loss_none(outputs, topn_label))
            cost = loss1 - loss2
        elif mode == 'topn2':
            outputs_victim_arr = outputs_victim.detach().cpu().numpy()
            topn_max_labels = torch.zeros(outputs.shape[0], topn).to(outputs.device)
            for j in range(topn):
                # ys外的最高概率类别输出值最大化
                if j == 0:
                    max_index = ys.detach().cpu().numpy()
                    min_value = np.min(outputs_victim_arr, axis=1)
                else:
                    max_index = np.argmax(outputs_victim_arr, axis=1)
                    min_value = np.min(outputs_victim_arr, axis=1)
                for i in range(len(max_index)):
                    outputs_victim_arr[i, max_index[i]] = min_value[i]
                topn_labels = np.argmax(outputs_victim_arr, axis=1)
                topn_labels = torch.tensor(topn_labels).to(outputs.device)
                topn_max_labels[:, j] = topn_labels
            topn_max_labels = topn_max_labels.to(torch.int64)
            loss1 = self.loss(outputs, ys)
            ws = F.softmax(torch.tensor([[5., 4., 3., 2., 1.]]).to(outputs.device), dim=1)
            loss2 = torch.tensor(0.).to(outputs.device)
            for i in range(topn_max_labels.shape[1]):
                topn_label = topn_max_labels[:, i]
                loss2 += torch.mean(ws[0,i]*self.loss_none(outputs, topn_label)/topn)
            cost = loss1 - loss2

        return cost
