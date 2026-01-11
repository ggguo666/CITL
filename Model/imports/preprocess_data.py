import os
import os
import warnings
import glob
import csv
import re
import numpy as np
from collections import OrderedDict
warnings.filterwarnings("ignore")

# Input data variables
phenotype = '/home/user/data/gsj/ABIDE_116/abide_116/Phenotypic_V1_0b_preprocessed1.csv'

# # ADHD
phenotype3 = '/home/user/data/gsj/ADHD/adhd200_preprocessed_phenotypics.tsv'

#
phenotype2 = '/home/user/data/gsj/ABIDE2/ABIDEII.csv'
data_folder = '/home/user/data/gsj/abide160/2222/stage1correlation/raw'
# phenotype = 'data/Phenotypic_V1_0b_preprocessed1.csv'

phenotype4 = '/home/user/data/gsj/ABIDE_116/abide_116/similar_label.csv'

def get_ids_ADHDno(path):
    ids = []
    # print(data)
    # 使用glob匹配所有.npy文件
    files = glob.glob(path + '*.npy')
    print(files)
    for file in files:
        # 从文件名提取ID
        id = os.path.basename(file).replace('sub-', '').replace('.npy', '')
        ids.append(id)
    ids.sort()
    # print('ids')
    return ids

def get_ADHD_score(subject_list, score):
    scores_dict = {}
    print(subject_list)
    with open(phenotype3, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            for subject_id in subject_list:
                folder_name = row['ID']
                name = folder_name.replace('sub-', '')
                if str(name) == subject_id:
                    # 获取标签并将其作为字典的键，值为 row[score]
                    scores_dict[subject_id] = int(row[score]) % 2
    sorted_data = dict(sorted(scores_dict.items()))
    print(sorted_data)
    return list(sorted_data.values())

# def get_label_dfc(subject_list, score):
#     scores_dict = {}
#     with open(phenotype) as csv_file:
#         reader = csv.DictReader(csv_file)
#         for row in reader:
#             for subject_id in subject_list:
#                 subject = extract_folder_number(subject_id)
#                 if int(row['SUB_ID']) == subject:
#                     print(11)
#                     scores_dict[subject_id] = int(row[score]) % 2
#
#     # 返回字典中所有值的列表
#     print(scores_dict)
#     return list(scores_dict.values())


def get_label_dfc(subject_list, score):
    # print(subject_list)
    scores_dict = {}

    # 使用字典来存储 phenotype 数据，减少重复查找
    phenotype_data = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            subject_id = row['SUB_ID']
            phenotype_data[subject_id] = int(row[score]) % 2  # 存储每个 subject_id 对应的 score

    # 去重处理 subject_list（如果可能有重复项）
    subject_set = set(subject_list)

    # print(subject_set)
    # 提取 subject_id 并获取对应标签
    for subject_id in subject_set:
        subject = extract_folder_number(subject_id)  # 假设这是有效的提取函数
        if subject in phenotype_data:
            scores_dict[subject_id] = phenotype_data[subject]
    sorted_dict = {k: scores_dict[k] for k in sorted(scores_dict, key=sort_key)}
    # print(sorted_dict)
    # 返回字典中所有值的列表
    return list(sorted_dict.values())


def sort_key_abide_adhd(key):
    prefix, suffix = key.split('_')
    return (prefix, int(suffix))
def get_label_dfc_abide_adhd(subject_list, score):
    scores_dict = {}
    csv1 = "/home/user/data/gsj/ABIDE_116/abide_116/dfc/abide2-abide1output.csv"
    # csv1 = "/home/caorui/g/data/adhd200_116/dfc_cnn/dfc/output.csv"
    # 将 subject_list 转换为集合，提高查找效率
    subject_set = set(map(str, subject_list))  # 这里假设 subject_list 中的元素需要转换为字符串

    with open(csv1) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # 直接检查 SUB_ID 是否在 subject_set 中
            if str(row['SUB_ID']) in subject_set:
                subject_id = str(row['SUB_ID'])
                scores_dict[subject_id] = int(row[score]) % 2
    sorted_dict = {k: scores_dict[k] for k in sorted(scores_dict, key=sort_key_abide_adhd)}
    # 返回字典中所有值的列表
    # print(sorted_dict)
    return list(sorted_dict.values())


def get_label_dfc_abide_adhd2(subject_list, score):
    scores_dict = {}
    csv1 = "/home/user/data/gsj/ADHD/116/adhd200_116/dfc/output.csv"
    # 将 subject_list 转换为集合，提高查找效率
    subject_set = set(map(str, subject_list))  # 这里假设 subject_list 中的元素需要转换为字符串
    # print(subject_set)
    with open(csv1) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # 直接检查 SUB_ID 是否在 subject_set 中
            if str(row['SUB_ID']) in subject_set:
                # print(1)
                subject_id = str(row['SUB_ID'])
                scores_dict[subject_id] = int(row[score]) % 2
    sorted_dict = {k: scores_dict[k] for k in sorted(scores_dict, key=sort_key_abide_adhd)}
    # 返回字典中所有值的列表
    # print(sorted_dict)
    return list(sorted_dict.values())


def sort_key_similar(key):
    prefix, suffix = key.split('_')
    return (prefix, int(suffix))
def get_label_dfc_similar(subject_list, score):
    scores_dict = {}
    with open(phenotype4) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            name = row['Filename'].replace('sub-00', '')
            # print(name)
            for subject_id in subject_list:
                # print(subject_id)
                if str(name) == str(subject_id):
                    # print(1)
                    scores_dict[subject_id] = int(row[score]) % 2

    # # 返回字典中所有值的列表
    # print(scores_dict)
    # 按照前缀和后缀共同排序
    sorted_dict = {k: scores_dict[k] for k in sorted(scores_dict, key=sort_key_similar)}
    print(sorted_dict)
    return list(scores_dict.values())

def get_ids(data):
    ids = []
    # print(data)
    # 使用glob匹配所有.npy文件
    files = glob.glob(data + '*.npy')
    # print(files)
    for file in files:
        # 从文件名提取ID
        id = int(os.path.basename(file).replace('sub-00', '').replace('.npy', ''))
        ids.append(id)
    ids.sort()
    print(ids)
    return ids

# 定义排序函数
def sort_paths_by_last_number(paths):
    # 提取路径中最后一部分的数字部分并转换为整数，然后按照数字大小排序
    sorted_paths = sorted(paths, key=lambda x: int(re.search(r'sub-(\d+)$', x).group(1)))
    return sorted_paths


def get_ids_ADHD(data_folder):
    ids = []
    # 使用glob匹配所有.npy文件夹
    folders = glob.glob(os.path.join(data_folder, 'sub-*'))
    sorted_paths = sort_paths_by_last_number(folders)
    # print()
    for folder in sorted_paths:
        folder_name = os.path.basename(folder)
        files = [file for file in glob.glob(os.path.join(folder, '*.npy')) if os.path.basename(file).startswith(tuple(map(str, range(10))))]
        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for file in files:
            # 从文件名提取ID
            file_name = os.path.basename(file)
            id = "{}-{}".format(folder_name.replace('sub-', ''), file_name.replace('.npy', ''))
            ids.append(id)
    # print(ids)
    return ids
def get_ids_abide2(data_folder):
    ids = []
    # 使用glob匹配所有.npy文件夹
    folders = glob.glob(os.path.join(data_folder, 'sub-*'))
    sorted_paths = sort_paths_by_last_number(folders)
    # print()
    for folder in sorted_paths:
        folder_name = os.path.basename(folder)
        files = [file for file in glob.glob(os.path.join(folder, '*.npy')) if
                 os.path.basename(file).startswith(tuple(map(str, range(10))))]
        # files = glob.glob(os.path.join(folder, '*.npy'))  # 按照文件名的最后部分（.npy）进行排序
        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for file in files:
            # 从文件名提取ID
            file_name = os.path.basename(file)
            id = "{}-{}".format(folder_name.replace('sub-', ''), file_name.replace('.npy', ''))
            ids.append(id)

    return ids

def get_ids_dfc_similar(data_folder):
    ids = []
    # 使用glob匹配所有.npy文件夹
    folders = glob.glob(os.path.join(data_folder, 'sub-00*'))

    sorted_paths = sort_paths_by_last_number(folders)
    # print()
    for folder in sorted_paths:
        folder_name = os.path.basename(folder)
        files = glob.glob(os.path.join(folder, '*.npy'))        # 按照文件名的最后部分（.npy）进行排序

        files = [path for path in files if re.search(r'\d+\.npy$', path)]
        # print(files)

        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for file in files:
            # 从文件名提取ID
            file_name = os.path.basename(file)
            id = "{}_{}".format(folder_name.replace('sub-00', ''), file_name.replace('.npy', ''))
            ids.append(id)

    return ids
def get_ids_dfc_abide_adhd(data_folder):
    ids = []
    # 使用glob匹配所有.npy文件夹
    folders = glob.glob(os.path.join(data_folder, 'sub-00*'))

    sorted_paths = sort_paths_by_last_number(folders)
    # print()
    for folder in sorted_paths:
        folder_name = os.path.basename(folder)
        files = glob.glob(os.path.join(folder, '*.npy'))        # 按照文件名的最后部分（.npy）进行排序

        files = [path for path in files if re.search(r'\d+\.npy$', path)]
        # print(files)

        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for file in files:
            # 从文件名提取ID
            file_name = os.path.basename(file)
            id = "{}_{}".format(folder_name.replace('sub-00', 'sub-00'), file_name.replace('.npy', ''))
            ids.append(id)
    return ids

def get_ids_dfc_abide_adhd2(data_folder):
    ids = []
    # 使用glob匹配所有.npy文件夹
    folders = glob.glob(os.path.join(data_folder, 'sub-*'))
    sorted_paths = sort_paths_by_last_number(folders)
    # print()
    for folder in sorted_paths:
        folder_name = os.path.basename(folder)
        files = [file for file in glob.glob(os.path.join(folder, '*.npy')) if os.path.basename(file).startswith(tuple(map(str, range(10))))]
        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for file in files:
            # 从文件名提取ID
            file_name = os.path.basename(file)
            id = "{}_{}".format(folder_name.replace('sub-', 'sub-'), file_name.replace('.npy', ''))
            ids.append(id)
    # print(ids)
    return ids

def get_ids_dfc(data_folder):
    ids = []
    # 使用glob匹配所有.npy文件夹
    folders = glob.glob(os.path.join(data_folder, 'sub-00*'))

    sorted_paths = sort_paths_by_last_number(folders)
    # print()
    for folder in sorted_paths:
        folder_name = os.path.basename(folder)
        files = glob.glob(os.path.join(folder, '*.npy'))        # 按照文件名的最后部分（.npy）进行排序

        files = [path for path in files if re.search(r'\d+\.npy$', path)]
        # print(files)

        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for file in files:
            # 从文件名提取ID
            file_name = os.path.basename(file)
            id = "{}-{}".format(folder_name.replace('sub-00', ''), file_name.replace('.npy', ''))
            ids.append(id)

    return ids

def get_ids_ave(data_folder):
    dirs = os.listdir(data_folder)
    ids = sorted(dirs)
    return ids


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if int(row['SUB_ID']) in subject_list:
                scores_dict[int(row['SUB_ID'])] = int(row[score]) % 2
    print(scores_dict)
    return list(scores_dict.values())

# Get phenotype values for a list of subjects
def extract_folder_number(subject_id):
    # 从格式为 '50002-0' 的条目中提取数字部分
    return str(subject_id.split('-')[0])

def sort_key(key):
    prefix, suffix = key.split('-')
    return (prefix, int(suffix))




def get_subject_score_ADHD(subject_list, score):
    # 对 subject_list 进行一次排序
    subject_list = sorted(subject_list, key=lambda x: (x.split('-')[0], int(x.split('-')[1])))
    print(len(subject_list))
    # 预先将 phenotype3 文件加载到内存中
    with open(phenotype3, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        # 创建一个以 subject ID 为键的字典，以减少查找时间
        data_dict = {row['ID'].replace('sub-', ''): row for row in reader}

    # 存储分数的字典
    scores_dict = {}

    # 遍历 subject_list 中的每个 subject_id
    for subject_id in subject_list:
        # 提取 subject_id 数字部分
        subject = extract_folder_number(subject_id)

        # 检查 subject 是否在 data_dict 中
        if subject in data_dict:
            row = data_dict[str(subject)]
            # 将 score 计算结果存入字典
            scores_dict[subject_id] = int(row[score]) % 2
        else:
            print(subject_id)
    # 按照自定义的排序规则进行排序
    sorted_dict = {k: scores_dict[k] for k in sorted(scores_dict, key=sort_key)}

    # 输出排序后的字典
    # print(sorted_dict)

    # 返回排序后的分数列表
    return list(sorted_dict.values())


# def get_subject_score_ADHD(subject_list, score):
#     scores_dict = {}
#     with open(phenotype3, newline='') as csv_file:
#         reader = csv.DictReader(csv_file, delimiter='\t')
#         for row in reader:
#             subject_list = sorted(subject_list, key=lambda x: (x.split('-')[0], int(x.split('-')[1])))
#             # print(subject_list)
#             # break
#             for subject_id in subject_list:
#                 subject = extract_folder_number(subject_id)
#                 # 检查是否有匹配的数字在 subject_list 中
#                 folder_name = row['ID']
#                 name = folder_name.replace('sub-', '')
#                 # print(name)
#                 if int(name) == subject:
#                     # 获取标签并将其作为字典的键，值为 row[score]
#                     scores_dict[subject_id] = int(row[score]) % 2
#     # 按照前缀和后缀共同排序
#     sorted_dict = {k: scores_dict[k] for k in sorted(scores_dict, key=sort_key)}
#     print(sorted_dict)
#     return list(sorted_dict.values())

def get_subject_score_dfc(subject_list, score):
    scores_dict = {}
    print(subject_list)
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for subject_id in subject_list:
                subject = extract_folder_number(subject_id)
                # 检查是否有匹配的数字在 subject_list 中
                if int(row['SUB_ID']) == subject:
                    # 获取标签并将其作为字典的键，值为 row[score]
                    scores_dict[subject_id] = row[score]

    return scores_dict
# def get_abide2_dfc(subject_list, score):
#     scores_dict = {}
#     # print(subject_list)
#     with open(phenotype2) as csv_file:
#         reader = csv.DictReader(csv_file)
#         for row in reader:
#             subject_list = sorted(subject_list, key=lambda x: (x.split('-')[0], int(x.split('-')[1])))
#             for subject_id in subject_list:
#                 subject = extract_folder_number(subject_id)
#                 # 检查是否有匹配的数字在 subject_list 中
#                 if int(row['SUB_ID']) == subject:
#                     # 获取标签并将其作为字典的键，值为 row[score]
#                     scores_dict[subject_id] = int(row[score]) % 2
#     # sorted_data = dict(sorted(scores_dict.items()))
#     # return sorted_data
#     # # 按照前缀和后缀共同排序
#     sorted_dict = {k: scores_dict[k] for k in sorted(scores_dict, key=sort_key)}
#     print(sorted_dict)
#     return list(sorted_dict.values())
def get_abide2_dfc(subject_list, score):
    # print(subject_list)
    scores_dict = {}

    # 使用字典来存储 phenotype 数据，减少重复查找
    phenotype_data = {}

    with open(phenotype2) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            subject_id = row['SUB_ID']
            phenotype_data[subject_id] = int(row[score]) % 2  # 存储每个 subject_id 对应的 score

    # 去重处理 subject_list（如果可能有重复项）
    subject_set = set(subject_list)

    # print(subject_set)
    # 提取 subject_id 并获取对应标签
    for subject_id in subject_set:
        subject = extract_folder_number(subject_id)  # 假设这是有效的提取函数
        if subject in phenotype_data:
            scores_dict[subject_id] = phenotype_data[subject]
    sorted_dict = {k: scores_dict[k] for k in sorted(scores_dict, key=sort_key)}
    # print(sorted_dict)
    # 返回字典中所有值的列表
    return list(sorted_dict.values())


def get_ABIDE_subject_score_ave(subject_list, score):
    scores_dict = {}
    # print(subject_list)
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:

            for subject_id in subject_list:
                subject = subject_id.replace('sub-00', '')
                # print(subject)
                # 检查是否有匹配的数字在 subject_list 中
                if int(row['SUB_ID']) == int(subject):
                    # print(int(row['SUB_ID']))
                    # print(subject)
                    # 获取标签并将其作为字典的键，值为 row[score]
                    scores_dict[subject_id] = int(row[score]) % 2


    return list(scores_dict.values())



def get_ABIDE2_subject_score_ave(subject_list, score):
    scores_dict = {}
    # print(len(subject_list))
    with open(phenotype2) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for subject_id in subject_list:
                # subject = extract_folder_number(subject_id)
                subject = subject_id.replace('sub-', '')
                # 检查是否有匹配的数字在 subject_list 中
                if int(row['SUB_ID']) == int(subject):
                    # 获取标签并将其作为字典的键，值为 row[score]
                    scores_dict[subject_id] = int(row[score]) % 2
    #sorted_data = dict(sorted(scores_dict.items()))
    #return sorted_data
    return list(scores_dict.values())




def get_subject_score_ave(subject_list, score):
    scores_dict = {}
    with open(phenotype3, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            for subject_id in subject_list:
                # subject = extract_folder_number(subject_id)
                subject = subject_id.replace('sub-', '')
                # 检查是否有匹配的数字在 subject_list 中
                folder_name = row['ID']
                name = folder_name.replace('sub-', '')
                # print(name)
                if int(name) == int(subject):
                    # 获取标签并将其作为字典的键，值为 row[score]
                    scores_dict[subject_id] = int(row[score]) % 2
                    # 按照前缀和后缀共同排序
    sorted_data = dict(sorted(scores_dict.items()))
    print(sorted_data)
    return list(sorted_data.values())




def get_networks(subject_list, data_folder):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder + "sub-00" +str(subject) + ".npy")
        matrix = np.load(fl)
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    # idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    # vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.stack(norm_networks)

    return matrix


# 文件名排序函数
def sort_filenames(filenames):
    # 提取文件名中的数字作为排序的关键字
    def extract_num(filename):
        return int(re.search(r'\d+', filename).group())


    sorted_filenames = sorted(filenames, key=extract_num)
    return sorted_filenames


def get_networks_dfc(data_folder):

    all_networks = []
    matrix = np.load(data_folder)
    all_networks.append(matrix)

    # idx = np.triu_indices_from(all_networks[0], 1)
    # norm_networks = [np.arctanh(mat) for mat in all_networks]
    # vec_networks = [mat[idx] for mat in norm_networks]
    # matrix = np.stack(norm_networks)
    return all_networks

