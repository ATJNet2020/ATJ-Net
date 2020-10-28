# -*- coding: future_fstrings -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import os
import gc
import CONSTANT
import numpy as np
import pandas as pd
import lightgbm as lgb
from hyperopt import hp
from sklearn.metrics import roc_auc_score

from model import BaseModel
from hyper import bayes_opt

class LGBMModel(BaseModel):

  def __init__(self, database, output_dir, kfold=10, keval=5, **kargs):
    super(LGBMModel, self).__init__(database, output_dir, kfold, keval)
    self.kargs = kargs

    n = database.main_table.cat_feat.shape[1]
    cat_feat = pd.DataFrame(
        database.main_table.cat_feat,
        columns=list(map(str, range(n))))
    m = database.main_table.num_feat.shape[1]
    num_feat = pd.DataFrame(
        database.main_table.num_feat,
        columns=list(map(str, range(n, n+m))))
    self.data = pd.concat([cat_feat, num_feat], axis=1)
    self.categorical_feature = list(map(str, range(n)))
    del cat_feat, num_feat
    gc.collect()

  def _train(self):
    aucs = []
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'num_threads': CONSTANT.JOBS,
        }
    param.update(self.kargs)
    for keval, train_idx, valid_idx in self.kFold():
      train_data = lgb.Dataset(
          self.data.iloc[train_idx], label=self.label[train_idx],
          categorical_feature=self.categorical_feature)
      valid_data = lgb.Dataset(
          self.data.iloc[valid_idx], label=self.label[valid_idx],
          categorical_feature=self.categorical_feature)
      bst = lgb.train(
          param, train_data, valid_sets=[valid_data],
          early_stopping_rounds=5, verbose_eval=False)
      with open(os.path.join(self.output_dir, f'save_model_{keval}.pt'), 'w') as f:
        f.write(bst.model_to_string(bst.best_iteration))
      ypred = bst.predict(
          self.data.iloc[valid_idx], num_iteration=bst.best_iteration)
      aucs.append(
          roc_auc_score(y_true=self.label[valid_idx], y_score=ypred))
    return np.array(aucs)
  
  def _predict(self):
    test_data = self.data.iloc[self.test_idx]
    test_label = self.label[self.test_idx]
    aucs, preds = [], []
    for keval in range(self.keval):
      with open(os.path.join(self.output_dir, f'save_model_{keval}.pt'), 'r') as f:
        bst = lgb.Booster(model_str=f.read())
      ypred = bst.predict(test_data, num_iteration=bst.best_iteration)
      aucs.append(roc_auc_score(y_true=test_label, y_score=ypred))
      preds.append(ypred)
    return np.array(aucs), np.stack(preds, axis=0)


def lgbm_default_param(database, output_dir, keval=5):
  model = LGBMModel(database, output_dir, keval=keval)
  model.train()
  model.predict()
  stat = model.stat()
  del model
  return stat

def lgbm_bayes_opt(database, output_dir, keval=5):
  space = {
      # 'class_weight': hp.choice('class_weight', [None, 'balanced']),
      'boosting_type': hp.choice('boosting_type', [
        {'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
        {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
        {'boosting_type': 'goss', 'subsample': 1.0}]),
      'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
      'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
      'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
      'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
      'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
      'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
      'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
      }
  def param_func(param):
    subsample = param['boosting_type'].get('subsample', 1.0)
    param['boosting_type'] = param['boosting_type']['boosting_type']
    param['subsample'] = subsample
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
      param[parameter_name] = int(param[parameter_name])

  best_param = bayes_opt(
      database, output_dir, LGBMModel, space, param_func=param_func)
  print('best_param:', best_param)

  model = LGBMModel(database, output_dir, keval=keval, **best_param)
  model.train()
  model.predict()
  stat = model.stat()
  del model
  return stat
