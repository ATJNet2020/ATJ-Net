# -*- coding: future_fstrings -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import numpy as np
import networkx as nx
from hyperopt import hp

import torch
import torch.nn as nn
import torch.nn.functional as F

import CONSTANT
from model import NNModel
from hyper import bayes_opt
from index import fetch_key_index, safe_index, arange_keyindex


class Identity(nn.Module):
  def __init__(self, *args, **kwargs):
    super(Identity, self).__init__()

  def forward(self, input):
    return input


class GNNNode(nn.Module):

  def __init__(
      self, table, key, bigraph, gnn_nodes,
      depth, level, emb_layers, random_arch, random_state,
      init_func=nn.init.xavier_normal_, activation=F.relu,
      emb_dropout=0.04, num_batch_norm=True,
      agg_mode='mean', agg_dropout=0.01):
    super().__init__()
    self.table = table
    self.name = table.name
    self.device = torch.device(CONSTANT.DEVICE)
    self.init_func = init_func
    self.act = activation
    self.agg_mode = agg_mode
    self.depth = depth

    if key is None or (table.id is not None and table.id.name == key):
      self.is_id = True
      self.key = table.id
      self.is_one = True
    else:
      self.is_id = False
      self.key = table.columns[key]
      self.is_one = self.key.keyindex.is_one

    self.no_of_cat_feat, self.no_of_num_feat = 0, 0
    if table.cat_feat.shape[0] != 0:
      self.emb_layers = [emb_layers[str(col.block_id)] for col in table.cat_columns]
      self.no_of_cat_feat = sum([e.weight.shape[1] for e in self.emb_layers])
      self.emb_dropout_layer = nn.Dropout(emb_dropout)

    if table.num_feat.shape[0] != 0:
      self.no_of_num_feat = table.num_feat.shape[1]
      self.num_bn_layer = nn.BatchNorm1d(self.no_of_num_feat) \
          if num_batch_norm else Identity()

    self.output_dims = self.no_of_cat_feat + self.no_of_num_feat
    self.sons = nn.ModuleDict()

    if depth < level:
      self.cat_columns_names = [c.name for c in self.table.cat_columns]
      for bid in bigraph.neighbors(self.name):
        for source_col_gname in bigraph.adj[self.name][bid].keys():
          son = nn.ModuleDict()
          for tname in bigraph.neighbors(bid):
            for target_col_gname in bigraph.adj[bid][tname].keys():
              cname = target_col_gname.split(':')[1]
              target_table = table.database.tables[tname]
              if tname == self.name:
                if target_table.id is not None and target_table.id.name == cname:
                  continue
                if target_table.columns[cname].keyindex.is_one: continue

              if (depth+1, tname) in gnn_nodes:
                grand = gnn_nodes[(depth+1, tname)]
              elif random_arch and random_state.rand() > 0.5:
                continue
              else:
                grand = GNNNode(
                    target_table, cname, bigraph, gnn_nodes,
                    depth+1, level, emb_layers, random_arch, random_state,
                    init_func, activation,
                    emb_dropout, num_batch_norm,
                    agg_mode, agg_dropout)
              self.output_dims += grand.output_dims
              son[target_col_gname] = grand
          self.sons[source_col_gname] = son

    if not self.is_id and not self.is_one:
      size = 32
      self.output_layer = nn.Linear(self.output_dims, size)
      self.init_func(self.output_layer.weight.data)
      self.output_dims = size
      self.output_drop_layer = nn.Dropout(agg_dropout)
    gnn_nodes[(depth, self.name)] = self


  def forward(self, raw_idx):
    assert self.no_of_cat_feat + self.no_of_num_feat != 0
    xs = []

    if self.is_id:
      idx = raw_idx
    else:
      key_index = fetch_key_index(self.key.keyindex, raw_idx, 20)
      idx = key_index.index + 1

    if CONSTANT.DEBUG and self.training:
      self.table.debug_lookup(idx-1)
      import pdb
      pdb.set_trace()

    if self.no_of_cat_feat != 0:
      cate_data_np = safe_index(self.table.cat_feat, idx-1)
      cate_data = torch.tensor(
          cate_data_np, dtype=torch.long, device=self.device)
      x = [self.emb_layers[i](cate_data[:, i])
           for i in range(cate_data_np.shape[1])]
      x = torch.cat(x, 1)
      xs.append(self.emb_dropout_layer(x))

    if self.no_of_num_feat != 0:
      cont_data_np = safe_index(self.table.num_feat, idx-1)
      cont_data = torch.tensor(
          cont_data_np, dtype=torch.float32, device=self.device)
      if cont_data.nelement() != 0:
        cont_data = self.num_bn_layer(cont_data)
      xs.append(cont_data)

    for source_col_gname, son in sorted(self.sons.items(), key=lambda x: x[0]):
      source_cname = source_col_gname.split(':')[1]
      if self.table.columns[source_cname].type == 'id':
        key_idx = idx
      else:
        key_idx = cate_data_np[:, self.cat_columns_names.index(source_cname)]
      for target_col_gname, grand in sorted(son.items(), key=lambda x: x[0]):
        xs.append(grand(key_idx))

    x = torch.cat(xs, 1)
    if not self.is_id:
      if hasattr(self, 'output_layer'):
        x = self.act(self.output_layer(x))
        x = self.output_drop_layer(x)

      x = self.aggregate_by_embbag(x, key_index, self.agg_mode)
      # x = self.aggregate_by_arange(x, key_index, self.agg_mode)
    return x

  def aggregate_by_embbag(self, weight, key_index, mode):
    offsets = torch.tensor(
        key_index.start, dtype=torch.long, device=self.device)
    input = torch.arange(
        weight.shape[0], dtype=torch.long, device=self.device)
    x = F.embedding_bag(
        weight=weight, input=input, offsets=offsets, mode=mode)
    return x

  def aggregate_by_arange(self, weight, key_index, mode):
    idx = torch.tensor(
        arange_keyindex(key_index), dtype=torch.long, device=self.device) # [B, T]
    weight = torch.cat([
        torch.zeros((1, weight.shape[1]), dtype=torch.float32, device=self.device),
        weight], axis=0)
    idx_shape = idx.shape
    x = torch.index_select(weight, 0, (idx+1).flatten()) # [B, T, H]
    x = x.reshape([idx_shape[0], idx_shape[1], weight.shape[-1]])
    if mode == 'mean':
      return x.mean(1)
    elif mode == 'sum':
      return x.sum(1)
    elif mode == 'max':
      return x.max(1)


  def __str__(self, depth=0):
    ret = '\t'*depth*2 + self.name + '(%d) ' % self.depth
    if hasattr(self, 'output_layer'):
      ret += '*'
    ret += f'\t{self.output_dims} <- {self.no_of_cat_feat} + {self.no_of_num_feat}'
    for _, son in sorted(self.sons.items(), key=lambda x: x[0]):
      for _, grand in sorted(son.items(), key=lambda x: x[0]):
        ret += f' + {grand.output_dims}'
    ret += '\n'

    for bid, son in sorted(self.sons.items(), key=lambda x: x[0]):
      ret += '\t'*(depth*2+1) + str(bid) + '\n'
      for _, grand in sorted(son.items(), key=lambda x: x[0]):
        ret += grand.__str__(depth+1)
    return ret


class FeedForwardNN(nn.Module):
  """ Gnn expansion
   -> (linear -> relu -> batch_norm -> dropout) * layer
   -> linear
  """
  def __init__(
      self, database, level, random_arch, random_state,
      init_func='kaiming_normal', activation='relu',
      lin_layer_sizes=[128, 64], lin_layer_dropouts=[0.001, 0.01],
      num_batch_norm=True, lin_batch_norm=True,
      emb_size_adder=1, emb_dropout=0.04,
      agg_mode='mean', agg_dropout=0.01):
    super().__init__()
    self.database = database
    self.level = level,
    self.main_table = database.main_table
    self.device = torch.device(CONSTANT.DEVICE)

    # Init function
    if init_func == 'kaiming_normal':
      self.init_func = nn.init.kaiming_normal_
    elif init_func == 'kaiming_uniform':
      self.init_func = nn.init.kaiming_uniform_
    elif init_func == 'xavier_normal':
      self.init_func = nn.init.xavier_normal_
    elif init_func == 'xavier_uniform':
      self.init_func = nn.init.xavier_uniform_
    else:
      assert False, 'Unknown init_func in FeedForwardNN'

    # Activation function
    if activation == 'relu':
      self.act = F.relu
    elif activation == 'elu':
      self.act = F.elu
    elif activation == 'leaky_relu':
      self.act = F.leaky_relu
    else:
      assert False, 'Unknown activation in FeedForwardNN'

    # Embedding layers
    bids, cat_unique = [], []
    for bid, block in sorted(database.info['blocks'].items()):
      is_valid = False
      for col_gname in block['columns']:
        tname, cname = col_gname.split(':')
        if database.info['tables'][tname][cname] != 'id':
          is_valid = True
          break
      if is_valid:
        cat_unique.append(block['unique'] + 1)
      else:
        cat_unique.append(np.NAN)
      bids.append(str(bid))
    emb_dims = np.ceil(
        np.log2(cat_unique) + emb_size_adder).astype(np.int32)
    self.emb_layers = nn.ModuleDict(
        {bid: nn.Embedding(x, y) if not np.isnan(x) else None
         for bid, x, y in zip(bids, cat_unique, emb_dims)})

    # Auto-Table-Join layers
    bigraph = nx.MultiGraph()
    for bid, block in sorted(database.info['blocks'].items()):
      if len(block['columns']) > 1:
        for col_gname in block['columns']:
          tname, cname = col_gname.split(':')
          bigraph.add_edge(bid, tname, key=col_gname)
    gnn_nodes = {}

    self.nets = GNNNode(
        self.main_table, None, bigraph, gnn_nodes,
        0, level, self.emb_layers, random_arch, random_state,
        self.init_func, self.act,
        emb_dropout=emb_dropout, num_batch_norm=num_batch_norm,
        agg_mode=agg_mode, agg_dropout=agg_dropout)
    print(str(self.nets) + '=====')
    for k, v in sorted(gnn_nodes.items(), key=lambda x: x[0]):
      print(k, v.name)

    # Linear Layers
    assert len(lin_layer_sizes) >= 0
    assert len(lin_layer_sizes) == len(lin_layer_dropouts)
    self.lin_layers = nn.ModuleList(
        [nn.Linear(self.nets.output_dims, lin_layer_sizes[0])] +\
        [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i+1])
         for i in range(len(lin_layer_sizes) - 1)])
    for lin_layer in self.lin_layers:
      self.init_func(lin_layer.weight.data)

    # Output Layer
    self.output_layer = nn.Linear(lin_layer_sizes[-1], 1)
    self.init_func(self.output_layer.weight.data)

    # Batch Norm Layers
    self.bn_layers = nn.ModuleList(
        [nn.BatchNorm1d(size) if lin_batch_norm else Identity()
         for size in lin_layer_sizes])

    # Dropout Layers
    self.dropout_layers = nn.ModuleList(
        [nn.Dropout(size) for size in lin_layer_dropouts])

  def forward(self, idx):
    x = self.nets(idx + 1)

    for lin_layer, dropout_layer, bn_layer in\
        zip(self.lin_layers, self.dropout_layers, self.bn_layers):

      x = self.act(lin_layer(x))
      x = bn_layer(x)
      x = dropout_layer(x)

    x = self.output_layer(x)
    return x.squeeze()


class AtjModel(NNModel):

  def __init__(
      self, database, output_dir, level, random_arch,
      kfold=10, keval=5, major_sample_ratio=5,
      learning_rate=1e-3, optimizer='adam',
      train_batch_size=128, weight_decay=1e-5,
      valid_batch_size=512, max_epoch=100000,
      early_stopping_limit=20, eval_interval=100,
      init_func='kaiming_normal', activation='relu',
      lin_layer_sizes=[128, 64], lin_layer_dropouts=[0.001, 0.01],
      num_batch_norm=True, lin_batch_norm=True,
      emb_size_adder=1, emb_dropout=0.04,
      agg_mode='mean', agg_dropout=0.01):

    self.level = level
    self.random_arch = random_arch
    self.init_func = init_func
    self.activation = activation
    self.lin_layer_sizes = lin_layer_sizes
    self.lin_layer_dropouts = lin_layer_dropouts
    self.num_batch_norm = num_batch_norm
    self.lin_batch_norm = lin_batch_norm
    self.emb_size_adder = emb_size_adder
    self.emb_dropout = emb_dropout
    self.agg_mode = agg_mode
    self.agg_dropout = agg_dropout

    super(AtjModel, self).__init__(database, output_dir,
        kfold=kfold, keval=keval, major_sample_ratio=major_sample_ratio,
        learning_rate=learning_rate, optimizer=optimizer,
        train_batch_size=train_batch_size, weight_decay=weight_decay,
        valid_batch_size=valid_batch_size, max_epoch=max_epoch,
        early_stopping_limit=early_stopping_limit, eval_interval=eval_interval)


  def _build(self):
    return FeedForwardNN(
        self.database, self.level,
        random_arch=self.random_arch, random_state=self.random_state,
        init_func=self.init_func,
        activation=self.activation,
        lin_layer_sizes=self.lin_layer_sizes,
        lin_layer_dropouts=self.lin_layer_dropouts,
        num_batch_norm=self.num_batch_norm,
        lin_batch_norm=self.lin_batch_norm,
        emb_size_adder=self.emb_size_adder,
        emb_dropout=self.emb_dropout,
        agg_mode=self.agg_mode,
        agg_dropout=self.agg_dropout).to(self.device)


def _atj_default_param(database, output_dir, level, random_arch, keval=5):
  model = AtjModel(database, output_dir,
                   level=level, random_arch=random_arch,
                   keval=keval, num_batch_norm=False)
  model.train()
  model.predict()
  stat = model.stat()
  del model
  return stat


def _atj_bayes_opt(database, output_dir, level, random_arch, keval=5):
  space = {
      'level': level,
      'random_arch': random_arch,
      'major_sample_ratio': hp.quniform('major_sample_ratio', 2, 20, 2),
      'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(0.1)),
      'train_batch_size': hp.choice('train_batch_size', [32, 64, 128, 256, 512]),
      'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
      'init_func': hp.choice('init_func', ['kaiming_normal', 'kaiming_uniform',
                                           'xavier_normal', 'xavier_uniform']),
      'activation': hp.choice('activation', ['relu', 'elu', 'leaky_relu']),
      'lin_layer_sizes': hp.choice('lin_layer_sizes', [
                                   [128, 64], [256, 128], [64, 32]]),
      'lin_layer_dropouts': hp.choice('lin_layer_dropouts', [
                                   [0.001, 0.01], [0.001, 0.001], [0.01, 0.01], [0, 0]]),
      'num_batch_norm': False,
      'lin_batch_norm': hp.choice('lin_batch_norm', [True, False]),
      'emb_size_adder': hp.quniform('emb_size_adder', 1, 5, 1),
      'emb_dropout': hp.loguniform('emb_dropout', np.log(1e-6), np.log(0.1)),
      'agg_mode': hp.choice('agg_mode', ['sum', 'mean', 'max']),
      'agg_dropout': hp.loguniform('agg_dropout', np.log(1e-6), np.log(0.1)),
      }
  best_param = bayes_opt(database, output_dir, AtjModel, space)
  print('best_param:', best_param)
  return '', ''
  # model = AtjModel(database, output_dir, keval=keval, **best_param)
  # model.train()
  # model.predict()
  # stat = model.stat()
  # del model
  # return stat


def atj0_default_param(database, output_dir, keval=5):
  return _atj_default_param(database, output_dir, level=0, random_arch=False, keval=keval)

def atj1_default_param(database, output_dir, keval=5):
  return _atj_default_param(database, output_dir, level=1, random_arch=False, keval=keval)

def atj2_default_param(database, output_dir, keval=5):
  return _atj_default_param(database, output_dir, level=2, random_arch=False, keval=keval)

def atj0_bayes_opt(database, output_dir, keval=5):
  return _atj_bayes_opt(database, output_dir, level=0, random_arch=False, keval=keval)

def atj1_bayes_opt(database, output_dir, keval=5):
  return _atj_bayes_opt(database, output_dir, level=1, random_arch=False, keval=keval)

def atj2_bayes_opt(database, output_dir, keval=5):
  return _atj_bayes_opt(database, output_dir, level=2, random_arch=False, keval=keval)


def atj1_random_bayes_opt(database, output_dir, keval=5):
  return _atj_bayes_opt(database, output_dir, level=1, random_arch=True, keval=keval)

def atj2_random_bayes_opt(database, output_dir, keval=5):
  return _atj_bayes_opt(database, output_dir, level=2, random_arch=True, keval=keval)
