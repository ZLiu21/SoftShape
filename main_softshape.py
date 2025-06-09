import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from ts_utils import set_seed, build_loss, save_cls_result, build_dataset, get_all_datasets, evaluate_model
from data.preprocessing import normalize_per_series, fill_nan_value, normalize_train_val_test
from data.dataloader import UCRDataset
from data.shape_size_hyp import ucr_hyp_dict_shape_size
import argparse
from models.SoftShapeModel import SoftShapeNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Base setup
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='CBF', help='dataset(in ucr)')  # ACSF1 GunPoint
    parser.add_argument('--dataroot', type=str, default='/home/lz/UCRArchive_2018', help='path of UCR folder')
    parser.add_argument('--num_class', type=int, default=2, help='number of class')
    parser.add_argument('--normalize_way', type=str, default='single', help='single or train_set')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')
   
    # Model setup
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--sparse_rate', type=float, default=0.50, help='0.1, 0.3, or 0.7')
    parser.add_argument('--shape_size', type=int, default=8, help='16, 32, or 48')
    parser.add_argument('--shape_use_ratio', type=int, default=0, help='0 is False, 1 is True')
    parser.add_argument('--shape_ratio', type=float, default=0.1)
    parser.add_argument('--shape_stride', type=int, default=4, help='the patch stride')
    parser.add_argument('--moe_num_experts', type=int, default=8)
    parser.add_argument('--warm_up_epoch', type=int, default=150, help='50, 100 or 200')
    parser.add_argument('--moeloss_rate', type=float, default=0.001, help='0.01 or 0.001')

    # Training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--use_large_batch', type=int, default=1, help='1 is True, 0 is False') ## Larger batch size can run faster
    parser.add_argument('--epoch', type=int, default=500, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:3')

    # Result setup
    parser.add_argument('--save_dir', type=str, default='/home/lz/SoftShape/result')
    parser.add_argument('--save_csv_name', type=str, default='softshape_five_fold_split')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)
    
    sum_dataset, sum_target, num_class = build_dataset(args)
    
    args.num_class = num_class
    args.moe_num_experts = num_class
    args.seq_len = sum_dataset.shape[1]
    args.batch_size = int(min(sum_dataset.shape[0] * 0.6/10, 16))
    
    args.shape_size = ucr_hyp_dict_shape_size[args.dataset]['shape_size']
    args.shape_use_ratio = ucr_hyp_dict_shape_size[args.dataset]['shape_use_ratio']
    args.shape_ratio = ucr_hyp_dict_shape_size[args.dataset]['shape_ratio']
    
    print("Training dataset = ", args.dataset, ", shape_size = ", args.shape_size)
    
    if args.shape_use_ratio == 1:
        args.shape_size = int(args.seq_len * args.shape_ratio)
        if args.shape_size <= args.shape_stride:
            args.shape_stride = min(2, args.shape_size)
    
    if args.shape_stride > args.shape_size:
        args.shape_stride = args.shape_size

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target)
    
    if args.use_large_batch == 1:
        ''' 
        Our latest experiments show that setting a larger batch size can effectively improve the runtime speed 
        without degrading SoftShape's overall classification performance on the UCR 128 datasets.
        '''
        args.batch_size = 512
        if train_datasets[0].shape[0] < args.batch_size:
            args.batch_size = train_datasets[0].shape[0]
        
    loss = build_loss(args).to(device)
    model = SoftShapeNet(seq_len=args.seq_len, shape_size=args.shape_size, num_channels=1, emb_dim=args.emb_dim, sparse_rate=args.sparse_rate, 
                         depth=args.depth, num_experts=args.moe_num_experts, num_classes=args.num_class, stride=args.shape_stride)

    model = model.to(device)
    model_init_state = model.state_dict()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    test_accuracies = []
    train_time = 0.0
    
    for i, train_dataset in enumerate(train_datasets):
        t = time.time()
        model.load_state_dict(model_init_state)

        print('{} fold start training and evaluate'.format(i))
            
        train_target = train_targets[i]
        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        train_dataset = train_dataset[:, :, np.newaxis]
        val_dataset = val_dataset[:, :, np.newaxis]
        test_dataset = test_dataset[:, :, np.newaxis]

        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)
        
        if args.normalize_way == 'single':
            # TODO normalize per series
            train_dataset = normalize_per_series(train_dataset)
            val_dataset = normalize_per_series(val_dataset)
            test_dataset = normalize_per_series(test_dataset)
        else:
            train_dataset, val_dataset, test_dataset = normalize_train_val_test(train_dataset, val_dataset,
                                                                                test_dataset)

        train_set = UCRDataset(torch.from_numpy(train_dataset).type(torch.FloatTensor).to(device),
                            torch.from_numpy(train_target).type(torch.FloatTensor).to(device).to(torch.int64))
        val_set = UCRDataset(torch.from_numpy(val_dataset).type(torch.FloatTensor).to(device),
                            torch.from_numpy(val_target).type(torch.FloatTensor).to(device).to(torch.int64))
        test_set = UCRDataset(torch.from_numpy(test_dataset).type(torch.FloatTensor).to(device),
                            torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64))

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

        last_loss = float('inf')
        stop_count = 0
        increase_count = 0

        min_val_loss = float('inf')
        test_accuracy = 0

        for epoch in range(args.epoch):

            if stop_count == 80 or increase_count == 80:
                print('model convergent at epoch {}, early stopping'.format(epoch))
                break

            epoch_train_loss = 0
            num_iterations = 0

            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                
                pred, moe_loss = model(x, num_epoch_i=epoch, warm_up_epoch=args.warm_up_epoch)
                step_loss = loss(pred, y)
                step_loss = step_loss + args.moeloss_rate * moe_loss
                step_loss.backward()
                optimizer.step()

                epoch_train_loss = epoch_train_loss + step_loss.item()
                num_iterations = num_iterations + 1

            epoch_train_loss = epoch_train_loss / num_iterations

            model.eval()
            val_loss, val_accu = evaluate_model(val_loader, model, loss)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                test_loss, test_accuracy = evaluate_model(test_loader, model, loss)

            if (epoch > args.warm_up_epoch) and (abs(last_loss - val_loss) <= 1e-4):
                stop_count = stop_count + 1
            else:
                stop_count = 0

            if (epoch > args.warm_up_epoch) and (val_loss > last_loss):
                increase_count = increase_count + 1
            else:
                increase_count = 0

            last_loss = val_loss

            if epoch % 100 == 0:
                print("epoch : {}, train loss: {:.4f}, test_accuracy: {:.4f}".format(epoch, epoch_train_loss, test_accuracy))

        test_accuracies.append(test_accuracy)
        t = time.time() - t
        train_time += t

        print('{} fold finish training.'.format(i))
        
    test_accuracies = torch.Tensor(test_accuracies).cpu().numpy()

    print("Training end. mean_test_acc: {:.4f}, training time {:.4f}".format(np.mean(test_accuracies), train_time))

    save_cls_result(args, np.mean(test_accuracies), train_time)

    print('Done!')
