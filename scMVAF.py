"""
Created on Thu Jan 11 14:33:42 2024

@author: lqx
"""
from preprocess import *  ####用于导入指定模块中的全部定义。
import scanpy as sc
from preprocess import generate_views
from ViewModel import MultiViewModel
import torch
import random
import warnings
import argparse
import scipy.io
import psutil
from time import time as get_time

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='scMVAF',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_name', default='deng')
    parser.add_argument('--view_num', default=9)
    parser.add_argument('--sample_ratio', default=0.9)
    parser.add_argument('--pretrain_epoch', default=90)
    parser.add_argument('--fit_epoch', default=150)
    parser.add_argument('--batch_size', default=256)  # 512
    parser.add_argument('--pretrain_lr', default=0.001)
    parser.add_argument('--fit_lr', default=0.001)
    parser.add_argument('--tol', default=0.001)
    args = parser.parse_args()

    data_name = args.data_name
    view_num = args.view_num

    arii = np.array([])
    nmii = np.array([])
    accc = np.array([])

    file_path = './datasets/%s.mat' % data_name  # 动态替换文件名
    mat_data = scipy.io.loadmat(file_path)

    # 加载 .mat 文件
    save_path = './model/%s.pth' % data_name
    # 提取 X 和 Y 矩阵
    x = mat_data['X']
    y = mat_data['Y'].T
    y = y.reshape(-1)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    n_clusters = adata.obs['Group'].unique().shape[0]
    print(n_clusters)
    sf = []
    views = generate_views(adata, args.view_num, args.sample_ratio)
    for k in range(args.view_num):
        print(views[k].shape)
    for i in range(args.view_num):
        views1 = sc.AnnData(views[i])
        adata_view = normalize(views1, filter_min_counts=False, size_factors=True,
                               normalize_input=False,
                               logtrans_input=False)
        # print(adata_view.X.shape)
        sf.append(adata_view.obs['size_factors'])
    print(sf[0].shape)
    print("=================== Init scMVAF ===================")
    model = MultiViewModel(
        z_dim=32,
        n_clusters=n_clusters,
        labels=y,
        viewNumber=args.view_num,
        size_factor=sf,
        batch_size=args.batch_size,
        encodeLayer=[256, 64],
        decodeLayer=[64, 256],
        views=views,
        lr=args.pretrain_lr,
        activation='relu',
        sigma=2.5,
        alpha=1.0,
        gamma=1.0,
        device='cuda')
    z_fusion = model.pretrain(views, args.pretrain_epoch, args.pretrain_lr)

    final_acc, final_nmi, final_ari, final_homo, final_comp = model.fit(z_fusion, X=x,
                                                                        input_size=x.shape[0],
                                                                        views=views,
                                                                        size_factor=sf,
                                                                        n_clusters=n_clusters,
                                                                        y=y, lr=args.fit_lr,
                                                                        epoch=args.fit_epoch,
                                                                        tol=args.tol,
                                                                        batch_size=args.batch_size,
                                                                        save_path=save_path, b=1)

    print(
        'Evaluating %s cells:ACC=%.4f, ARI= %.4f, NMI= %.4f' % (
            data_name, final_acc, final_ari, final_nmi))

    resulttips = ' : '
    # 打开文件，使用 'w' 模式表示写入文件
    txtview = 'result1.txt'

    with open(txtview, 'a') as file:
        # 将变量的值写入文件，并保留四位小数
        file.write(data_name + resulttips)
        file.write(
            "ACC {:.3f}, ARI {:.3f}, NMI {:.3f}".format(
                np.mean(accc), np.mean(arii), np.mean(nmii)))
        file.write("\n")
