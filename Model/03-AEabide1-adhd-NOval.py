import os
from os import listdir
import numpy as np
import os.path as osp
import argparse
import time
import copy
import torch
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from imports.cnn_Dataset import ADHD_ave_Dataset_official
from torch import nn
from net.cnn import CNN
from net.AE2abide_stage3 import Autoencoder
from imports.utils import train_test_split_stage3_adhd, train_test_split_stage3
from sklearn.model_selection import KFold
import shutil
import random
import pandas as pd
import torch.nn.functional as func
from matplotlib import pyplot as plt
import os
import re
import random
from sklearn.metrics import roc_auc_score


def extract_epoch_number(filename):
    # 正则表达式匹配最后一部分数字
    match = re.search(r'(\d+)(?=\.\w+$)', filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # 如果没有匹配到数字，返回一个默认值或者根据情况处理


def shangsanjaio(data):
    upper_triangular_data = []
    for matrix in data:
        matrix1 = torch.triu(matrix, diagonal=1)
        # print(matrix1)
        # 展平成一维向量
        flat_upper_triangular = torch.masked_select(matrix1,
                                                    torch.triu(torch.ones_like(matrix1), diagonal=1).bool())
        upper_triangular_data.append(flat_upper_triangular)
    return upper_triangular_data




device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 设置全局随机数种子 77
seed = 3
seed2 = 3
# seed = np.random.randint(0, 100)
# seed2 = np.random.randint(0, 100)
print("种子点:", seed)
print("数据划分种子点", seed2)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
# 如果使用GPU，也设置GPU上的随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def main():
    # 定义超参数
    input_dim = (116, 116)  # 输入维度
    hidden_dim = 116  # 隐藏层维度

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')  # 64
    parser.add_argument('--dataroot', type=str, default='/home/user/data/gsj/ADHD/116/adhd200_116/dfc/68ave',
                        help='root directory of the dataset')
    parser.add_argument('--fold', type=int, default=10, help='training which fold')

    parser.add_argument('--lr1', type=float, default=0.0002, help='learning rate')  # 0.000005
    parser.add_argument('--weightdecay1', type=float, default=0.0001, help='regularization')  # 0.005

    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weightdecay', type=float, default=0.00007, help='regularization')

    parser.add_argument('--stepsize', type=int, default=50, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.8, help='scheduler shrinking rate')


    parser.add_argument('--indim', type=int, default=116, help='feature dim')
    parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
    opt = parser.parse_args()

    #################### Parameter Initialization #######################
    path = opt.dataroot
    opt_method = opt.optim
    num_epoch = opt.n_epochs
    fold = opt.fold

    ################## Define Dataloader ##################################
    test_accuracyave = 0
    test_accuracy_list = []
    sen_list = []
    spe_list = []
    auc_list = []
    for fold in range(0, fold):
        # 加载数据集示例
        dataset = ADHD_ave_Dataset_official(path, save_dir=path)
        tr_index, te_index = train_test_split_stage3_adhd(fold=fold, seed=seed2)
        tr_index = list(tr_index)
        te_index = list(te_index)

        print("训练集样本数:", len(tr_index))
        print("测试集样本数:", len(te_index))

        # 创建训练集、验证集和测试集的数据集
        train_dataset = torch.utils.data.Subset(dataset, tr_index)
        test_dataset = torch.utils.data.Subset(dataset, te_index)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize)

        # 创建模型
        autoencoder = Autoencoder(input_dim, hidden_dim, device).to(device)
        classifier = CNN(0.5, 0, 32, 128, 96, 512, dim=opt.indim).to(device)
        # classifier = CNN(0.2, 0, 128, 256, 96, 512, dim=opt.indim).to(device)

        # 定义损失函数和优化器
        # criterion1 = nn.L1Loss().to(device)

        criterion1 = nn.CosineEmbeddingLoss().to(device)
        criterion2 = nn.CrossEntropyLoss().to(device)

        ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=opt.lr1, weight_decay=opt.weightdecay1)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)

        # scheduler1 = lr_scheduler.StepLR(ae_optimizer, step_size=opt.stepsize1, gamma=opt.gamma1)
        scheduler = lr_scheduler.StepLR(classifier_optimizer, step_size=opt.stepsize, gamma=opt.gamma)
        ###################### Network Training Function#####################################
        def train(loader):
            total_loss = 0.0
            total_loss_AE = 0.0
            autoencoder.train()
            classifier.train()
            pred = []
            label = []
            for datas, labels in loader:
                data, y = datas.to(device), labels.to(device)
                targets = torch.ones(len(y)).to(device)
                upper_triangular_data = shangsanjaio(data)
                upper_triangular_data = torch.stack(upper_triangular_data).to(device)

                # 训练 Autoencoder
                ae_optimizer.zero_grad()
                reconstructions, encoded_matrix = autoencoder(upper_triangular_data)
                #
                # reconstructions = shangsanjaio(reconstructions)
                # reconstructions = torch.stack(reconstructions).to(device)

                # print(reconstructions.shape)
                # print(upper_triangular_data.shape)

                ae_loss = criterion1(reconstructions, upper_triangular_data, targets)
                # print(ae_loss)
                # ae_loss = criterion1(reconstructions, upper_triangular_data)
                ae_loss.backward()
                ae_optimizer.step()

                # 提取特征并训练分类器
                decoded_data, encoded_data = autoencoder(upper_triangular_data)
                classifier_optimizer.zero_grad()

                output, regression = classifier(encoded_data)
                # print(output)
                class_loss = criterion2(output, y.to(torch.float64))  # 预测损失
                # print(class_loss)
                class_loss.backward()
                classifier_optimizer.step()

                total_loss = total_loss + ae_loss.item() + class_loss.item()
                total_loss_AE = total_loss_AE + ae_loss.item()

                pred.append(output.max(dim=1)[1])
                # print(pred)
                # y_list = y.tolist()
                label.append(y.max(dim=1)[1])
                # print(label)
            # print(pred)
            # scheduler1.step()
            scheduler.step()

            y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
            y_true = torch.cat(label, dim=0).cpu().detach().numpy()
            tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
            # epoch_sen = tp / (tp + fn)
            # epoch_spe = tn / (tn + fp)
            epoch_acc = (tn + tp) / (tn + tp + fn + fp)

            return epoch_acc, total_loss / len(train_dataset), total_loss_AE / len(train_dataset)

        def test(loader):
            autoencoder.eval()
            classifier.eval()
            pred = []
            label = []
            total_loss = 0
            auc_output = []
            with torch.no_grad():
                for datas, labels in loader:
                    data, y = datas.to(device), labels.to(device)
                    targets = torch.ones(len(y)).to(device)

                    upper_triangular_data = shangsanjaio(data)
                    upper_triangular_data = torch.stack(upper_triangular_data).to(device)

                    #######ae
                    reconstructions, encoded_matrix = autoencoder(upper_triangular_data)
                    # reconstructions = shangsanjaio(reconstructions)
                    # reconstructions = torch.stack(reconstructions).to(device)
                    # print(len(reconstructions))
                    ae_loss = criterion1(reconstructions, upper_triangular_data, targets)
                    # ae_loss = criterion1(reconstructions, upper_triangular_data)
                    # 提取特征并训练分类器
                    decoded_data, encoded_data = autoencoder(upper_triangular_data)
                    # 分类
                    output, regression = classifier(encoded_data)
                    # print(output)
                    class_loss = criterion2(output, y.to(torch.float64))  # 预测损失

                    total_loss = total_loss + ae_loss.item() + class_loss.item()
                    auc_output.append(output)

                    pred.append(output.max(dim=1)[1])
                    # print(pred)
                    # y_list = y.tolist()
                    label.append(y.max(dim=1)[1])

                y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
                # print(y_pred)
                y_true = torch.cat(label, dim=0).cpu().detach().numpy()

                auc_output = torch.cat(auc_output, dim=0).cpu().detach().numpy()
                # print(auc_output)
                auc = roc_auc_score(y_true, auc_output[:, 1])
                # auc = roc_auc_score(y_true, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
                epoch_sen = tp / (tp + fn)
                epoch_spe = tn / (tn + fp)
                epoch_acc = (tn + tp) / (tn + tp + fn + fp)
                return epoch_spe, epoch_sen, epoch_acc, auc, total_loss / len(test_dataset)

        #######################################################################################
        ############################   Model Training #########################################
        #######################################################################################
        test_accuracy1 = 0
        spe1 = 0
        sen1 = 0
        for epoch in range(0, num_epoch):
            since = time.time()
            tr_acc, tr_loss, loss_AE = train(train_loader)
            spe, sen, test_acc, auc, test_loss = test(test_loader)

            time_elapsed = time.time() - since
            print('*====**')
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Fold: {:03d},Epoch: {:03d}, Train Loss: {:.4f}, AE Loss: {:.4f} '
                  'Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, sen: {:.4f}, spe: {:.4f}, auc: {:.4f}'.format(fold, epoch,
                                                                                                            tr_loss,
                                                                                                            loss_AE,
                                                                                                            tr_acc,
                                                                                                            test_loss,
                                                                                                            test_acc,
                                                                                                            sen, spe, auc))

            if test_acc > test_accuracy1:
                test_accuracy1 = round(test_acc, 4)
                spe1 = round(spe, 4)
                sen1 = round(sen, 4)
                auc1 = round(auc, 4)
        print("========================================================================")
        print("Test Acc: {:.4f}".format(test_accuracy1))
        print("sen:{:.4f},spe:{:.4f},auc:{:.4f}".format(sen1, spe1, auc1))

        test_accuracyave = test_accuracyave + test_accuracy1
        test_accuracy_list.append(test_accuracy1)
        sen_list.append(sen1)
        spe_list.append(spe1)
        auc_list.append(auc1)
    print(test_accuracy_list)
    print(sen_list)
    print(spe_list)
    print(auc_list)
    print("Test aveAcc: {:.4f}".format(test_accuracyave / 10))


if __name__ == "__main__":
    main()