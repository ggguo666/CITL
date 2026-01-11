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
from torch_geometric.data import DataLoader
from imports.cnn_Dataset import ASD_Dataset_official
from torch import nn
from net.cnn import CNN
# from net.cnn2 import CNN
from imports.utils import train_val_test_split
from sklearn.model_selection import KFold
import shutil

import random
import pandas as pd
import torch.nn.functional as func
from matplotlib import pyplot as plt

import numpy as np
# 定义保存模型的文件夹路径
checkpoint_dir = "/home/user/data/gsj/111111111/my_cnn_model/BaseModel/"
# 创建一个文件夹用于保存模型

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 定义保存模型的函数
def save_checkpoint(epoch, fold, model, optimizer, loss):
    checkpoint_filename = f"checkpoint_fold{fold}+epoch_{epoch}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    # 保存模型的状态字典、优化器的状态字典等
    print("save")
    torch.save({
        'fold' :fold,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # 如果有其他需要保存的内容，可以在这里添加
    }, checkpoint_path)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

seed = np.random.randint(0, 100)
print("种子点:", seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
# 如果使用GPU，也设置GPU上的随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')#64
    parser.add_argument('--dataroot', type=str, default='/home/user/data/gsj/abide_116/abide_116/dfc', help='root directory of the dataset')
    parser.add_argument('--fold', type=int, default=10, help='training which fold')
    parser.add_argument('--lr', type = float, default=0.00005, help='learning rate')
    parser.add_argument('--weightdecay', type=float, default=0.005, help='regularization')
    parser.add_argument('--indim', type=int, default=116, help='feature dim')
    parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
    opt = parser.parse_args()

    #################### Parameter Initialization #######################
    path = opt.dataroot
    opt_method = opt.optim
    num_epoch = opt.n_epochs
    fold = opt.fold
    data_folder = "/home/user/data/gsj/abide_116/abide_116/dfc/cpac/filt_noglobal/raw"
    ################## Define Dataloader ##################################
    # data.x:节点的特征矩阵，大小为[num_nodes, num_node_features]。
    # data.edge_index图中的边的信息，采用COO格式记录，大小为[2, num_edges]，类型为torch.long。COO格式也就是CoordinateFormat，采用三元组进行存储，三元组内元素分别为行坐标、列坐标和元素值，此处没有元素值，所以只有2行，num_edges列，每一列表示一个元素。
    # data.edge_attr边的特征矩阵，大小为[num_edges, num_edge_features]
    # data.y训练目标，允许任意形状，比如节点级别的为[num_nodes, *]，图级别的为[1, *]
    # data.pos节点的位置矩阵，大小为[num_node, num_dimensions]
    # 创建五折交叉验证对象
    test_accuracyave = 0
    test_accuracy_list = []
    for fold in range(0, fold):
        # 加载数据集示例
        dataset = ASD_Dataset_official(path, save_dir=path)
        print(dataset)
        tr_index, val_index, te_index = train_val_test_split(data_folder)


        tr_index = list(tr_index)
        val_index = list(val_index)
        te_index = list(te_index)

        print("训练集样本数:", len(tr_index))
        print("验证集样本数:", len(val_index))
        print("测试集样本数:", len(te_index))

        # train_dataset = dataset[tr_index]
        # val_dataset = dataset[val_index]
        # test_dataset = dataset[te_index]
        # # print(train_dataset)
        # # print(val_dataset)
        # # print(test_dataset)
        #
        # train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=True)#len(val_dataset)
        # test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True)#len(test_dataset)


        # 创建训练集、验证集和测试集的数据集
        train_dataset = torch.utils.data.Subset(dataset, tr_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)
        test_dataset = torch.utils.data.Subset(dataset, te_index)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize)
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize)



        ############### Define Graph Deep Learning Network ##########################
        #特征维度 pooling ratio 类别

        model = CNN(0.5, 0, 32, 128, 96, 512, dim=opt.indim).to(device)
        # model = CNN(0.5, 0.4, 32, 128, 96, 512, dim=opt.indim).to(device)

        criterion_pa = nn.CrossEntropyLoss().to(device)
        print(model)

        if opt_method == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
        elif opt_method == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weightdecay, nesterov=True)


        ###################### Network Training Function#####################################
        def train(loader):
            model.train()
            pred = []
            label = []
            loss_all = 0
            for datas, labels in loader:
                data, y = datas.to(device), labels.to(device)

                optimizer.zero_grad()
                output, regression = model(data)
                # print(output)
                loss_pa = criterion_pa(output, y.to(torch.float64))  # 预测损失

                loss = loss_pa
                loss.backward()
                optimizer.step()
                loss_all += loss.item()

                pred.append(output.max(dim=1)[1])
                # print(pred)
                # y_list = y.tolist()
                label.append(y.max(dim=1)[1])
                # print(label)
            y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
            y_true = torch.cat(label, dim=0).cpu().detach().numpy()
            tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
            # epoch_sen = tp / (tp + fn)
            # epoch_spe = tn / (tn + fp)
            epoch_acc = (tn + tp) / (tn + tp + fn + fp)

            # return epoch_sen, epoch_spe, epoch_acc, loss_all / len(train_dataset)
            return epoch_acc, loss_all / len(train_dataset)

        def val(loader):
            model.eval()
            pred = []
            label = []
            loss_all = 0
            with torch.no_grad():
                for datas, labels in loader:
                    data, y = datas.to(device), labels.to(device)

                    output, regression = model(data)
                    loss = criterion_pa(output, y.to(torch.float64))  # 预测损失
                    loss_all += loss.item()

                    pred.append(output.max(dim=1)[1])
                    # print(pred)
                    # y_list = y.tolist()
                    label.append(y.max(dim=1)[1])

                y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
                y_true = torch.cat(label, dim=0).cpu().detach().numpy()
                tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
                # epoch_sen = tp / (tp + fn)
                # epoch_spe = tn / (tn + fp)
                epoch_acc = (tn + tp) / (tn + tp + fn + fp)
                return epoch_acc, loss_all / len(val_dataset)

        def test(loader):
            pred = []
            label = []
            loss_all = 0
            for datas, labels in loader:
                data, y = datas.to(device), labels.to(device)

                output, regression = model(data)
                loss = criterion_pa(output, y.to(torch.float64))  # 预测损失
                loss_all += loss.item()
                pred.append(output.max(dim=1)[1])
                # print(pred)
                # y_list = y.tolist()
                label.append(y.max(dim=1)[1])

            y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
            y_true = torch.cat(label, dim=0).cpu().detach().numpy()
            tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
            epoch_sen = tp / (tp + fn)
            epoch_spe = tn / (tn + fp)
            epoch_acc = (tn + tp) / (tn + tp + fn + fp)
            return epoch_spe, epoch_sen, epoch_acc, loss_all / len(test_dataset)
        #######################################################################################
        ############################   Model Training #########################################
        #######################################################################################

        for epoch in range(0, num_epoch):
            since = time.time()
            tr_acc, tr_loss = train(train_loader)
            val_acc, val_loss = val(val_loader)
            time_elapsed = time.time() - since
            print('*====**')
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Fold: {:03d},Epoch: {:03d}, Train Loss: {:.7f}, '
                  'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(fold, epoch, tr_loss,
                                                               tr_acc, val_loss, val_acc))


            save_checkpoint(epoch, fold, model, optimizer, val_loss)

    #######################################################################################
    ######################### Testing on testing set ######################################
    #######################################################################################

        # 定义保存模型的文件夹路径
        checkpoint_dir = "/home/user/data/gsj/111111111/my_cnn_model/BaseModel/"
        best_model_dir = "/home/user/data/gsj/111111111/my_cnn_model/Bset_BaseModel/"
        best_model_path = ""
        test_accuracy1 = 0
        test_ave = 0
        # 获取保存模型的文件夹中所有模型检查点的文件名
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if
                            f.endswith('.pth') and f.startswith(f"checkpoint_fold{fold}")]  # 仅选择特定fold的文件
        num_epoch = len(checkpoint_files)  # 假设每个epoch都有一个检查点文件
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            # 加载模型
            checkpoint = torch.load(checkpoint_path)  # , map_location=torch.device('cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            with torch.no_grad():
                sen, spe, test_accuracy, test_l = test(test_loader)
            test_ave = test_ave + test_accuracy
            if test_accuracy > test_accuracy1:
                test_accuracy1 = test_accuracy
                spe1 = spe
                sen1 = sen
                best_model_path = checkpoint_path
            # print("Test Acc: {:.7f}, Test loss: {:.7f}".format(test_accuracy, test_l))
            # print("sen:{},spe:{}".format(sen, spe))
        print("========================================================================")
        print("Test Acc: {:.7f}".format(test_accuracy1))
        print("sen:{},spe:{}".format(sen1, spe1))
        # 移动最佳模型到best_model_dir文件夹中
        if best_model_path:
            best_model_name = os.path.basename(best_model_path)
            best_model_dest = os.path.join(best_model_dir, best_model_name)
            os.makedirs(best_model_dir, exist_ok=True)
            shutil.move(best_model_path, best_model_dest)
            print(f"Best model moved to: {best_model_dest}")
        else:
            print("No best model found.")
        break
if __name__ == "__main__":
    main()