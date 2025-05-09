import random

import numpy as np
import torch

import os
import pandas as pd

from data.preprocessing import load_data, transfer_labels, k_fold
from models.loss import cross_entropy, reconstruction_loss
from sklearn.metrics import accuracy_score


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)
    sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target, num_classes


def build_loss(args):
    if args.loss == 'cross_entropy':
        return cross_entropy()
    elif args.loss == 'reconstruction':
        return reconstruction_loss()


def get_all_datasets(data, target):
    return k_fold(data, target)


def evaluate_model(val_loader, model, loss):
    val_loss = 0
    val_pred_labels = []
    real_labels = []

    sum_len = 0
    for data, target in val_loader:
        with torch.no_grad():
            val_pred, _ = model(data)
            val_loss += loss(val_pred, target).item()
            sum_len += len(target)
            val_pred_labels.append(torch.argmax(val_pred.data, axis=1).cpu().numpy())
            real_labels.append(target.cpu().numpy())

    val_pred_labels = np.concatenate(val_pred_labels)
    real_labels = np.concatenate(real_labels)

    return val_loss / sum_len, accuracy_score(real_labels, val_pred_labels)


def save_cls_result(args, mean_accu, train_time):
    save_path = os.path.join(args.save_dir, '', args.save_csv_name + '_cls.csv')
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path, index_col=0)
    else:
        result_form = pd.DataFrame(columns=['dataset_name', 'mean_accu', 'train_time'])

    result_form = pd.concat([result_form, pd.DataFrame([{'dataset_name': args.dataset, 'mean_accu': '%.4f' % mean_accu, 'train_time': '%.4f' % train_time}])], ignore_index=True)

    result_form.to_csv(save_path, index=True, index_label="id")

