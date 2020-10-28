# -*- coding: future_fstrings -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import os
import time
import copy
import torch
import random
import CONSTANT
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import time_human_format

class BaseModel(object):

  def __init__(self, database, output_dir, kfold=10, keval=5):
    self.database = database
    self.output_dir = output_dir
    self.kfold = kfold
    self.keval = keval

    os.makedirs(output_dir, exist_ok=True)
    self.set_seed()
    self.random_state = np.random.RandomState(1234)

    self.main_table = database.main_table
    self.n_lines = self.main_table.n_lines
    self.label = database.label.data

    if database.flag is None:
      all_index = np.arange(self.n_lines, dtype=np.int32)
      self.random_state.shuffle(all_index)
      self.train_idx = all_index[ : int(self.n_lines*0.8)]
      self.test_idx  = all_index[int(self.n_lines*0.8) : ]
    else:
      self.train_idx = np.where(database.flag.data != 0)[0].astype(np.int32)
      self.test_idx  = np.where(database.flag.data == 0)[0].astype(np.int32)


    if CONSTANT.DEBUG:
      if hasattr(self.main_table, 'debug_raw_idx'):
        raw_train_idx = self.main_table.debug_raw_idx[self.train_idx] + 1
        raw_test_idx  = self.main_table.debug_raw_idx[self.test_idx] + 1
      else:
        raw_train_idx, raw_test_idx = self.train_idx + 1, self.test_idx + 1

      with open(os.path.join(
          self.output_dir, database.name + '_train_line_id.txt'), 'w') as f:
        for v in raw_train_idx:
          print(v, file=f)
      with open(os.path.join(
          self.output_dir, database.name + '_test_line_id.txt'), 'w') as f:
        for v in raw_test_idx:
          print(v, file=f)

  def _train(self):
    raise NotImplementedError

  def train(self):
    start_time = time.time()
    auc = self._train()
    self.train_time_cost = time.time() - start_time
    return auc

  def _predict(self):
    raise NotImplementedError

  def predict(self):
    start_time = time.time()
    self.auc, preds = self._predict()
    self.predict_time_cost = time.time() - start_time
    return self.auc, preds

  def kFold(self):
    n = self.train_idx.shape[0]
    size = n // self.kfold + 1
    end = 0
    for keval in range(self.keval):
      start, end = end, min(end + size, n)
      valid_idx = self.train_idx[start : end]
      train_idx = np.concatenate(
          [self.train_idx[ : start], self.train_idx[end : ]],
          axis=0)
      yield keval, train_idx, valid_idx

  def stat(self):
    percent_auc = self.auc * 100
    t_number, t_metric = time_human_format(self.train_time_cost)
    p_number, p_metric = time_human_format(self.predict_time_cost)
    return percent_auc, 'Dataset: %s\tAUC: %.4f%% (%.2f%%)\tTrain Cost: %.1f%s\tPredict Cost: %.1f%s' % (
        self.database.name, percent_auc.mean(), percent_auc.std(), t_number, t_metric, p_number, p_metric)

  def set_seed(self, seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class NNModel(BaseModel):

  def __init__(
      self, database, output_dir,
      kfold=10, keval=5, major_sample_ratio=5,
      learning_rate=1e-3, optimizer='adam',
      train_batch_size=128, weight_decay=1e-5,
      valid_batch_size=512, max_epoch=100,
      early_stopping_limit=20, eval_interval=100):

    super(NNModel, self).__init__(database, output_dir, kfold=kfold, keval=keval)

    assert torch.cuda.is_available()
    self.device = torch.device(CONSTANT.DEVICE)
    self.model = self._build()

    self.criterion = torch.nn.BCEWithLogitsLoss()
    if optimizer == 'adam':
      self.optimizer = torch.optim.Adam(
          self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
      self.optimizer = torch.optim.SGD(
          self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    self.init_state = copy.deepcopy(self.model.state_dict())
    self.init_state_opt = copy.deepcopy(self.optimizer.state_dict())

    self.major_sample_ratio = major_sample_ratio
    self.train_batch_size = train_batch_size
    self.valid_batch_size = valid_batch_size
    self.max_epoch = max_epoch
    self.early_stopping_limit = early_stopping_limit
    self.eval_interval = eval_interval

  def _build(self):
    raise NotImplementedError

  def _train(self):
    aucs = []
    for keval, train_idx, valid_idx in self.kFold():
      step = 0
      trail = [0.0]
      break_flag = False
      early_stopping_count = 0

      self.model.load_state_dict(self.init_state)
      self.optimizer.load_state_dict(self.init_state_opt)

      batch_time_start = time.time()
      for epoch in range(self.max_epoch):
        down_sampled_train_idx = self.sample_majority(train_idx)
        for train_batch_idx in self.dataloader(
            down_sampled_train_idx, self.train_batch_size, True, True):

          if step % self.eval_interval == 0:
            self.model.eval()
            auc, _ = self.eval(valid_idx)
            self.model.train()
            batch_time_interval = time.time() - batch_time_start
            info_str = f'Keval %d Epoch %d Iter %d: AUC on validation: %.4f%%  Cost: %.2fs' % (
                    keval, epoch, step, auc*100, batch_time_interval)
            batch_time_start = time.time()

            if auc > max(trail):
              with open(os.path.join(self.output_dir, f'save_model_{keval}.pt'), 'wb') as f:
                torch.save(self.model.state_dict(), f)
              early_stopping_count = 0
              info_str += ' *'
            else:
              early_stopping_count += 1
            print(info_str)
            trail.append(auc)

          if early_stopping_count > self.early_stopping_limit:
            break_flag = True
            break

          pred = self.model(train_batch_idx)
          label = torch.tensor(
              self.label[train_batch_idx], dtype=torch.float32, device=self.device)
          loss = self.criterion(pred, label)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          step += 1
        if break_flag:
          break
      aucs.append(max(trail))
    return np.array(aucs)

  def _predict(self):
    aucs, preds = [], []
    for keval in range(self.keval):
      with open(os.path.join(self.output_dir, f'save_model_{keval}.pt'), 'rb') as f:
        self.model.load_state_dict(torch.load(f))
      auc, pred = self.eval(self.test_idx)
      aucs.append(auc)
      preds.append(pred)
    return np.array(aucs), np.stack(preds, axis=0)

  def sample_majority(self, idx):
    labels = self.label[idx]
    one_idx = idx[labels == 1]
    zero_idx = idx[labels == 0]
    if one_idx.shape[0] > zero_idx.shape[0] and \
       one_idx.shape[0] / zero_idx.shape[0] > self.major_sample_ratio:
      return np.concatenate([
        self.random_state.choice(
          one_idx, round(zero_idx.shape[0]*self.major_sample_ratio), False),
        zero_idx], axis=0)
    if zero_idx.shape[0] > one_idx.shape[0] and \
       zero_idx.shape[0] / one_idx.shape[0] > self.major_sample_ratio:
      return np.concatenate([
        self.random_state.choice(
          zero_idx, round(one_idx.shape[0]*self.major_sample_ratio), False),
        one_idx], axis=0)
    return idx

  def dataloader(self, idx, batch_size, shuffle=False, align=False):
    if shuffle:
      self.random_state.shuffle(idx)
    n = idx.shape[0]
    for start in range(0, n, batch_size):
      end = start + batch_size
      if end <= n:
        yield idx[start : end]
      elif align:
        yield np.concatenate(
            [ idx[start : ], self.random_state.choice(idx, end - n) ],
            axis=0)
      else:
        yield idx[start : ]

  def eval(self, valid_idx):
    preds = []
    for batch_idx in self.dataloader(valid_idx, self.valid_batch_size, False):
      pred = self.model(batch_idx)
      pred = torch.sigmoid(pred)
      preds.append(pred.cpu().data.numpy())
    preds = np.concatenate(preds, axis=0)
    labels = self.label[valid_idx]
    auc = roc_auc_score(y_true=labels, y_score=preds)
    return auc, preds
