# Copyright 2017 Goekcen Eraslan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle, os, numbers

import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import torch

def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    # norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    # assert 'n_count' not in adata.obs, norm_error

    # if adata.X.size < 50e6: # check if adata.X is integer only if array is small
    #     if sp.sparse.issparse(adata.X):
    #         assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
    #     else:
    #         assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata

def sample_columns(matrix, sample_ratio):
    num_cols = matrix.shape[1]
    num_sample_cols = int(num_cols * sample_ratio)

    # 生成随机索引
    random_indices = torch.randperm(num_cols)[:num_sample_cols]

    # 根据随机索引选择列
    sampled_matrix = matrix[:, random_indices]

    return sampled_matrix

def generate_views(matrix, num_views, sample_ratio):
    views = []
    for _ in range(num_views):
        sampled_view = sample_columns(matrix.X, sample_ratio)
        views.append(sampled_view)
    return views

def normalize(adata, filter_min_counts=False, size_factors=True, normalize_input=False, logtrans_input=False):


    if size_factors:
        # sc.pp.normalize_total(adata)
        # adata.obs['n_counts'] = adata.X.sum(axis=1)
        #
        # # 计算大小因子
        # adata.obs['size_factors'] = adata.obs['n_counts'] / np.median(adata.obs['n_counts'])
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

