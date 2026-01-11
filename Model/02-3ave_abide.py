import os
import torch
import numpy as np

def compute_average_feature(folder_path):
    dirs = os.listdir(folder_path)
    dirs = sorted(dirs)

    # 遍历文件夹中的文件
    for d1 in dirs:

        path1 = os.path.join(folder_path, d1)

        output_dir1 = os.path.join('/home/user/data/gsj/ABIDE_116/abide_116/dfc/abide2-abide1ave', d1)
        os.makedirs(output_dir1, exist_ok=True)  # 确保文件夹存在


        len_path = os.path.join("/home/user/data/gsj/ABIDE_116/abide_116/dfc/",d1)

        total_len = len(os.listdir(len_path))-1
        print(total_len)
        for d2 in os.listdir(path1):
            path2 = os.path.join(path1, d2)
            name0 = os.listdir(path2)
            len_0 = len(name0)
            if len_0 >= total_len/2:
                print(d2)
                output_dir2 = os.path.join(output_dir1, d2)
                os.makedirs(output_dir2, exist_ok=True)  # 确保文件夹存在
                sum_of_features = None
                total_feature_matrices = 0
                for name in os.listdir(path2):
                    file_pathx = os.path.join(path2, name)
                    features = torch.load(file_pathx)
                    features = features[0][0]
                    features_array = features.cpu().numpy()
                    # 如果是第一个特征矩阵，初始化 sum_of_features
                    if sum_of_features is None:
                        # 如果是第一个特征矩阵，初始化 sum_of_features
                        sum_of_features = features_array
                        print(sum_of_features)
                    else:
                        # 累加特征矩阵
                        sum_of_features += features_array

                    total_feature_matrices += 1
                print(sum_of_features.shape)
                print(total_feature_matrices)
                # 计算平均特征矩阵
                average_features = sum_of_features / total_feature_matrices
                print(average_features)
                # 保存平均特征矩阵到文件
                output_file_path = os.path.join(output_dir2, "average_features.npy")
                np.save(output_file_path, average_features)
                if total_feature_matrices == 0:
                    print("文件夹中没有特征文件，无法计算平均特征值。")
                else:
                    print(f"已计算并保存 {total_feature_matrices} 个平均特征值文件到目录: {output_file_path}")
                break



def main():
    # 假设保存特征文件的文件夹路径
    folder_path = '/home/user/data/gsj/ABIDE_116/abide_116/dfc/abide2-abide1-67afterstage2'

    # 计算平均特征值
    compute_average_feature(folder_path)

if __name__ == "__main__":
    main()