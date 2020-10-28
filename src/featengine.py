import os
import time
import json
import copy
import logging
import numpy as np
import pandas as pd
from table import Table
from index import build_key_index, moment_group_by
from utils import parallel, is_nan_column, is_dup_column

class Feat:

  def __init__(self, engine):
    self.engine = engine

  def fit_transform(self, database):
    raise NotImplementedError

  def gen_feat_name(self, feat_name, feat_type):
    cls_name = self.__class__.__name__
    prefix = feat_type[0]
    return '%s%s(%s)' % (prefix, cls_name, feat_name)


class LocateUniqueColumnToId(Feat):
  """ 定位每张表中，具有完全 unique 值的列作为 Id 列
      如果有多于两列满足条件，则删掉多余列并提示警告
      如果已有 Id 列，则仅检查该列是否满足条件，不再考虑其他列唯一性
  """
  def fit_transform(self, database):
    for table in sorted(database.tables.values()):
      id_col = table.id
      if id_col is None:
        todo_list = table.cat_columns
        def func(col):
          return pd.Series(col.data).nunique(dropna=True)
        n_unique = np.array(parallel(func, todo_list), dtype=np.int32)
        idx = np.where(n_unique == table.n_lines)[0]
        if len(idx) == 1:
          table.info[todo_list[idx[0]].name] = 'id'
        elif len(idx) > 1:
          for col in todo_list[idx[1:]]:
            table.drop_column(col.name)
          logging.warning(
              'More than one column in table %s have unique values: %s' % \
                  (table.name, todo_list[idx][0].global_name)) + \
                  ', we have dropped the unnecessary.'
          table.info[todo_list[idx[0]].name] = 'id'
      else:
        assert pd.Series(id_col.data).nunique(dropna=True) == table.n_lines, \
            '%s is not an id column' % id_col.global_name


def multicat2table_pd(id_col, multi_col, sep):
  # Build new table with [id, multi-cat]
  sub = pd.DataFrame(data={id_col.name: id_col.data, multi_col.name: multi_col.data})
  # Split multi-cat column to a new table with [cat, order]
  flat = sub[multi_col.name].str.split(sep, expand=True).stack().reset_index(level=1)
  flat.columns = ['_order', multi_col.name]
  flat['_order'] += 1
  # Join [id] and [cat, order] by index
  sub = sub.drop([multi_col.name], axis=1).join(flat, how='inner')
  sub = sub.astype(
      {id_col.name: str, '_order': np.uint32, multi_col.name: str}, copy=False)
  sub.reset_index()
  return sub

def multicat2table_ar(id_col, multi_col, sep):
  def foo(x):
    if isinstance(x, str):
      arr = tuple(x.split(sep))
      return arr, len(arr)
    else:
      return tuple(), 0
  arr, cnt = np.vectorize(foo, otypes=[tuple, np.int32])(multi_col.data)
  m, n = id_col.data.shape[0], cnt.sum()

  sub_id = np.empty(n, dtype=id_col.data.dtype)
  sub_multi = np.empty(n, dtype=multi_col.data.dtype)
  sub_order = np.empty(n, dtype=np.uint32)

  cur = 0
  for i in range(m):
    for j in range(cnt[i]):
      sub_id[cur] = id_col.data[i]
      sub_multi[cur] = arr[i][j]
      sub_order[cur] = j+1
      cur += 1
  assert cur == n

  return pd.DataFrame(data={
    id_col.name: sub_id,
    '_order': sub_order,
    multi_col.name: sub_multi})


class SplitMulticat(Feat):
  """ 将 multi-cat 属性拆分为两张表，每个表都只包含 cat 属性
  """
  def fit_transform(self, database):
    k = len(database.tables)
    for table in list(sorted(database.tables.values())):
      # Prepare multi-cat columns
      todo_list = table.multicat_columns
      if len(todo_list) == 0:
        continue

      # Locate Id Column
      id_col = table.id
      if id_col is None:
        id_data = np.arange(1, table.n_lines+1, dtype=np.int32)
        id_col = table.add_column('_id', 'id', id_data.astype(str))

      for col in todo_list:
        # Parse seperator from info
        sep = col.type.split(':', 1)
        if len(sep) == 1:
          sep = ','
        else:
          sep = sep[1]

        # Drop old multi-cat column
        table.drop_column(col.name, ignore_block_manager=True)

        # sub = multicat2table_pd(id_col, col, sep)
        sub = multicat2table_ar(id_col, col, sep)

        # Add the new table [id, cat, order]
        assert f'table_{k}' not in database.info['tables']
        database.info['tables'][f'table_{k}'] = {
            id_col.name: 'cat',
            '_order': 'order',
            col.name: 'cat'}
        database.tables[f'table_{k}'] = Table(database, f'table_{k}', sub)

        # Modify BlockManager
        bid = database.block_manager.col2block[f'{table.name}:{col.name}']
        database.block_manager.blocks[bid]['columns'].remove(f'{table.name}:{col.name}')
        database.block_manager.blocks[bid]['columns'].append(f'table_{k}:{col.name}')
        database.block_manager.blocks[bid]['columns'].sort()
        del database.block_manager.col2block[f'{table.name}:{col.name}']
        database.block_manager.col2block[f'table_{k}:{col.name}'] = bid

        bid = database.block_manager.col2block[f'{table.name}:{id_col.name}']
        database.block_manager.col2block[f'table_{k}:{id_col.name}'] = bid
        database.block_manager.blocks[bid]['columns'].append(f'table_{k}:{id_col.name}')
        database.block_manager.blocks[bid]['columns'].sort()
        k += 1


class RemapCategorical(Feat):

  def fit_transform(self, database):
    """ 将离散属性映射为连续的 id
        空值映射为 0，有效值从 1 开始计数，按照频数降序排列
    """
    database.block_manager.reset_freq()


class IdColumnData:
  def __init__(self, n_lines):
    self.shape = (n_lines, )
    self.size = 1
    self.itemsize = 8
    self.dtype = np.int32

class RearrangeIdColumn(Feat):

  def fit_transform(self, database):
    """ id 列需要满足的条件是: 该列覆盖了所在 block 内的所有值，且从 1-N 顺序排列
        根据 id 列对表排序，使得整个表可以被 id 列直接索引
    """
    for table in list(sorted(database.tables.values())):
      # Re-assign id and cat columns
      for col in table.idcat_columns:
        if table.n_lines == col.block['unique']:
          database.info['tables'][table.name][col.name] = 'id'
        else:
          database.info['tables'][table.name][col.name] = 'cat'

      # Continue if there is no id column
      id_col = table.id
      if id_col is None:
        continue

      # Re-arrange dataframe by id column
      idx = id_col.data.argsort()
      for col in table.columns.values():
        col.data = col.data[idx]
      assert np.all(id_col.data == np.arange(
             1, id_col.block['unique']+1, dtype=np.int32)), \
             id_col.global_name

      n_lines = id_col.data.shape[0]
      del id_col.data
      id_col.data = IdColumnData(n_lines)


class IndexCategorical(Feat):
  """ 对所有离散属性列构造索引
      去掉所有表的 order 列
  """
  def fit_transform(self, database):
    # Build keyindex
    def func(col):
      order = col.table.order
      order_data = order.data if order is not None else None
      # TODO
      # set_time = True if order is not None and order.type == 'time' else False
      set_time = False
      col.keyindex = build_key_index(
          col.data, order_data, col.block['unique'], set_time)
    parallel(func, database.cat_columns)

    # Remove all order attributes
    for table in database.tables.values():
      order = table.order
      if order is not None and order.type == 'order':
        table.drop_column(order.name)


class UpdateCache(Feat):
  """ 更新 database.cache_columns 到表
  """
  def fit_transform(self, database):
    self.engine.update_cache_columns()


class TimeNum(Feat):
  """ 记录所有出现过的时间的最小值和最大值
      增加时间列，年月日星期时分新列
  """
  def fit_transform(self, database):
    for table in sorted(database.tables.values()):
      for col in table.time_columns:
        col_data = pd.Series(col.data)
        new_data = (col_data.astype(np.int64) / 10**9).astype(np.float32).to_numpy()
        new_data = np.where(new_data >= 0, new_data, np.NaN)
        self.engine.cache_column(table.name, self.gen_feat_name(col.name, 'num'), 'num', new_data)

        self.engine.cache_column(table.name, f'nTimeYear:{col.name}',
            'num', col_data.dt.year.values.astype(np.float32))
        self.engine.cache_column(table.name, f'nTimeMonth:{col.name}',
            'num', col_data.dt.month.values.astype(np.float32))
        self.engine.cache_column(table.name, f'nTimeDay:{col.name}',
            'num', col_data.dt.day.values.astype(np.float32))
        self.engine.cache_column(table.name, f'nTimeWeekday:{col.name}',
            'num', col_data.dt.weekday.values.astype(np.float32))
        self.engine.cache_column(table.name, f'nTimeHour:{col.name}',
            'num', col_data.dt.hour.values.astype(np.float32))
        self.engine.cache_column(table.name, f'nTimeMinute:{col.name}',
            'num', col_data.dt.minute.values.astype(np.float32))


class Freq(Feat):
  """ 计算每个 many 离散列在表内的频数
  """
  def fit_transform(self, database):
    for col in database.cat_columns:
      if not col.keyindex.is_one:
        size = col.keyindex.end - col.keyindex.start
        size = size.astype(np.float32, copy=False)
        size[0] = np.nan
        freq_data = size[col.data]
        self.engine.cache_column(
            col.table.name, self.gen_feat_name(col.name, 'num'), 'num', freq_data)


class FreqGroupBy(Feat):
  """ tA:aA 和 tB:aB 是两个可以 Join 的离散属性
      tB:aB 的出现基数是 many
      则我们计算 a1 对应了几个 a2
  """
  def fit_transform(self, database):
    info = database.info['tables']
    for block in database.block_manager.blocks.values():
      for A in block['columns']:
        for B in block['columns']:
          tA, aA = A.split(':')
          tB, aB = B.split(':')
          colA = database.tables[tA][aA]
          colB = database.tables[tB][aB]
          if A != B and colB.type != 'id' and not colB.keyindex.is_one:
            size = colB.keyindex.end - colB.keyindex.start
            size = size.astype(np.float32, copy=False)
            size[0] = np.nan
            if colA.type == 'id':
              freq_data = size[ : colA.n_lines]
            else:
              freq_data = size[colA.data]
            self.engine.cache_column(tA, f'nFreq({B})GroupBy({A})', 'num', freq_data)


class MomentGroupByInside(Feat):
  """ a 是个 many 离散属性，n 是一个连续属性
      我们针对 a 计算 n 的均值、方差、偏度、峰度
  """
  def fit_transform(self, database):
    todo_list = []
    for table in database.tables.values():
      for cat_col in table.cat_columns:
        for num_col in table.num_columns:
          if not cat_col.keyindex.is_one:
            todo_list.append((cat_col, num_col))

    def func(args):
      cat_col, num_col = args
      mean, std, skew, kurt = moment_group_by(
          cat_col.keyindex, num_col.data)
      mean[0] = std[0] = skew[0] = kurt[0] = np.nan
      return mean[cat_col.data], std[cat_col.data], \
             skew[cat_col.data], kurt[cat_col.data]
    rets = parallel(func, todo_list)

    for (cat_col, num_col), (mean, std, skew, kurt) in zip(todo_list, rets):
      self.engine.cache_column(
          cat_col.table.name, f'nMean({num_col.name})GroupBy({cat_col.name})', 'num', mean)
      self.engine.cache_column(
          cat_col.table.name, f'nStd({num_col.name})GroupBy({cat_col.name})', 'num', std)
      self.engine.cache_column(
          cat_col.table.name, f'nSkew({num_col.name})GroupBy({cat_col.name})', 'num', skew)
      self.engine.cache_column(
          cat_col.table.name, f'nKurt({num_col.name})GroupBy({cat_col.name})', 'num', kurt)


class DropNan(Feat):
  """ 扔掉缺失率大于一定数值的列
  """
  def fit_transform(self, database):
    todo_list = database.attr_columns
    def func(col):
      return is_nan_column(col.type, col.data)
    rets = parallel(func, todo_list)
    for col, is_drop in zip(todo_list, rets):
      if is_drop:
        col.table.drop_column(col.name)


class DropDup(Feat):
  """ 扔掉重复率大于一定数值的列
      采样提高效率
  """
  def fit_transform(self, database):
    todo_list = database.attr_columns
    def func(col):
      return is_dup_column(col.type, col.data)
    rets = parallel(func, todo_list)
    for col, is_drop in zip(todo_list, rets):
      if is_drop:
        col.table.drop_column(col.name)


class StandardScaler(Feat):
  """ 对 num 列进行 z-score 标准化, 不增加新列
      并填充 NaN
  """
  def fit_transform(self, database):
    def func(col):
      mean, std = np.nanmean(col.data), np.nanstd(col.data)
      col.data = (col.data - mean) / std
      np.nan_to_num(col.data, copy=False)
    parallel(func, database.num_columns)


class NanToNum(Feat):
  """ 填充 NaN """
  def fit_transform(self, database):
    def func(col):
      np.nan_to_num(col.data, copy=False)
    parallel(func, database.num_columns)


class BasicFeatPipeline:
  def __init__(self):
    self.order = [
        LocateUniqueColumnToId,
        SplitMulticat,
        RemapCategorical,
        RearrangeIdColumn,
    ]

class SimpleFeatPipeline:
  def __init__(self):
    self.order = [
        LocateUniqueColumnToId,
        SplitMulticat,
        RemapCategorical,
        RearrangeIdColumn,
        IndexCategorical,
        DropNan,
        DropDup,
        StandardScaler,
    ]

class ProperFeatPipeline:
  def __init__(self):
    self.order = [
        LocateUniqueColumnToId,
        SplitMulticat,
        RemapCategorical,
        RearrangeIdColumn,
        IndexCategorical,
        DropNan,
        DropDup,
        TimeNum,
        Freq,
        FreqGroupBy,
        UpdateCache,
        StandardScaler,
    ]

class FullFeatPipeline:
  def __init__(self):
    self.order = [
        LocateUniqueColumnToId,
        SplitMulticat,
        RemapCategorical,
        RearrangeIdColumn,
        IndexCategorical,
        DropNan,
        DropDup,
        TimeNum,
        Freq,
        FreqGroupBy,
        MomentGroupByInside,
        UpdateCache,
        StandardScaler,
    ]

class ComplexFeatPipeline:
  def __init__(self):
    self.order = [
        LocateUniqueColumnToId,
        SplitMulticat,
        RemapCategorical,
        RearrangeIdColumn,
        IndexCategorical,
        DropNan,
        DropDup,
        TimeNum, UpdateCache,
        Freq, UpdateCache,
        FreqGroupBy, UpdateCache,
        MomentGroupByInside, UpdateCache,
        StandardScaler,
    ]

class TestFeatPipeline:
  def __init__(self):
    self.order = [
        LocateUniqueColumnToId,
        SplitMulticat,
        RemapCategorical,
        RearrangeIdColumn,
        IndexCategorical,
        # TimeNum, UpdateCache,
        Freq, UpdateCache,
        FreqGroupBy, UpdateCache,
        MomentGroupByInside, UpdateCache,
        DropNan,
        DropDup,
        StandardScaler,
    ]


class FeatEngine:
  def __init__(self, feat_pipeline):
    self.feat_pipeline = feat_pipeline
    self._cache_columns = {}

  def fit_transform(self, database):
    self.feats_order = []
    print('Feature engineering by %s' \
        % (self.feat_pipeline.__class__.__name__))
    self.database = database
    total_start_time = time.time()
    for feat_cls in self.feat_pipeline.order:
      start_time = time.time()
      feat = feat_cls(self)
      feat.fit_transform(database)
      self.feats_order.append(feat)
      print('\t%s\t%.2fs'
          % (feat.__class__.__name__, time.time() - start_time))
    del self.database
    print('Feature engineering, cost %.1fs' % (time.time() - total_start_time))

  def cache_column(self, tname, attr, typ, data):
    """ 向表中新增一列
        目前只支持新增 num 属性列
    """
    assert typ == 'num' and data.dtype == np.float32, \
        'Only support add num columns to a table.'
    assert data.shape[0] == self.database.tables[tname].n_lines, \
        '%s:%s n_lines not match.' % (tname, attr)

    assert attr not in self.database.info['tables'][tname], \
        '%s:%s already exists.' % (tname, attr)
    assert (tname, attr) not in self._cache_columns, \
        '%s:%s already exists in cache columns' % (tname, attr)
    self._cache_columns[(tname, attr)] = (typ, data)

  def update_cache_columns(self):
    """ 更新 cache_columns 到表
        筛除 NaN 过多和重复值过多的列
        目前只支持新增 num 属性列
    """
    def is_drop(args):
      return True if is_nan_column(*args) or is_dup_column(*args) else False
      # return False
    self._cache_columns = [(k, v) for k, v in self._cache_columns.items()]
    self._cache_columns.sort()
    ret = parallel(is_drop, [v for k, v in self._cache_columns])

    rest_table2data = {}
    for ((tname, attr), (typ, data)), drop_flag in zip(self._cache_columns, ret):
      if not drop_flag:
        self.database.tables[tname].add_column(attr, typ, data)
    self._cache_columns = {}
