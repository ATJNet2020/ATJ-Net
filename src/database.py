import os
import gc
import copy
import json
import time
import argparse
import numpy as np
import pandas as pd
import prettytable as pt

from utils import *
from featengine import *
from table import Table
from block import BlockManager


class Database:

  def __init__(self, database_path, output_path=None):
    """ 从一个路径加载数据库

        路径中应包含 info.json 文件，它描述了多表数据的源信息。其有效格式如下:
        sep:       csv 文件的列分隔符。可选，默认为 \t
        task:      目标任务。可选，默认为 classification
        label:     目标列。必选，格式为 '表名:列名'，对于 linkprediction 列名可为空
        time_col:  主时间列名（可选）
        tables:    { 表名: {属性名: 属性类型} }
                   指明了表的列名及其属性，列名不应以'_'开头，列属性的有效格式如下:
                   id:        id属性    能唯一标识每行的列  指定可以节约推断时间，不必选
                   cat:       离散属性  不能比大小的属性，如职业、商品类目、标签
                   num:       数值属性  可以比大小的属性，如年龄、统计值、各类整数、浮点数等
                   time:      时间属性  格式为 time:[s|ms|us|ns]时间戳或有效的 datetime 时间字符串，默认为 ms
                   flag:      标记属性  可选，1表示训练集，0表示测试集。若不选，则另需要一个 主表_test.xxx 的文件
                   multi-cat: 多值属性  格式为 multi-cat:分隔符 默认为逗号
        relations: [ [ {'表名:属性名'} ] ]
                   指明了离散列之间的 joinable 关系

        其他文件应包含 info 中描述的所有表的 csv 数据文件
        文件名格式为 '表名.xxx'，第一行必须为表头，指明列名
        假设 info 中描述了 N 个表，则路径下至少有 N 个数据文件
        若 info 中指明了 flag 列，说明文件已经划分了训练测试数据；否则另需要一个 '主表_test.xxx' 的文件
    """
    database_path = database_path.rstrip('/')
    print('Loading raw databases from', database_path)
    start_time = time.time()
    self.name = os.path.basename(database_path)
    self.database_path = database_path
    self.output_path = output_path
    self._load_info()
    self._load_data()
    self.cache_columns = {}
    print('Finish loading database %s, cost %.2fs' % (
      self.name, time.time() - start_time))

  # ============================== 初始化加载函数 ================================

  def _load_info(self):
    """ 从路径下读取并解析 info.json 文件 """
    with open(os.path.join(self.database_path, 'info.json')) as f:
      info = json.load(f)

    # Check csv seperator
    if 'sep' not in info:
      info['sep'] = '\t'
    else:
      assert info['sep'] is not '\n'

    # Check task
    if 'task' not in info:
      info['task'] = 'classification'
    else:
      info['task'] = info['task'].lower()
      assert info['task'] in ['classification', 'regression', 'linkprediction'], \
          'The task must be in ["classification", "regression", "linkprediction"].'

    # Check label
    assert 'label' in info, 'info["label"] must be specified.'
    tname, attr = info['label'].split(':')
    if info['task'][0] == 'l':
      assert len(attr) == 0, \
          'label %s:%s is not needed for link prediction.' % (tname, attr)
    else:
      assert tname in info['tables'] and attr in info['tables'][tname], \
          'info["label"] (%s) not in table schema' % info['label']
      assert info['tables'][tname][attr] == 'num', 'The label must be numerical.'

    # Check flag
    flag = None
    for tname, tschema in info['tables'].items():
      for attr, typ in tschema.items():
        if typ == 'flag':
          assert flag is None
          flag = tname + ':' + attr
    if flag is not None:
      assert flag.split(':')[0] == info['label'].split(':')[0]

    # Check time_col
    if 'time_col' in info:
      for table in info['tables'].values():
        for attr, typ in table.items():
          if attr == info['time_col']:
            assert typ.startswith('time')

    # Check schema
    for tname, tschema in info['tables'].items():
      assert len(tschema) > 0, 'tschema must have one attr.'
      for attr, typ in tschema.items():
        assert typ in ['id', 'cat', 'num', 'flag'] \
            or typ.startswith('multi-cat') or typ.startswith('time') \
            or typ.startswith('emb'), \
            '%s:%s %s is not an valid type.' % (tname, attr, typ)

    self.info = info
    self.block_manager = BlockManager(self)
    del self.info['relations']

  def _load_data(self):
    """ 从路径下读取并解析数据文件 """
    info = self.info

    # Add table schema for main_test
    main_table = info['label'].split(':')[0]
    if 'flag' not in info['tables'][main_table].values():
      info['tables'][main_table + '_test'] = info['tables'][main_table]

    # Prepare candidate filenames
    filenames = os.listdir(self.database_path)
    filenames.remove('info.json')
    filenames = {os.path.splitext(os.path.basename(fname))[0] : fname
        for fname in filenames}

    # Build type2dtype map
    type2dtype = {
        'num': np.float32,
        'id': str,
        'cat': str,
        'multi-cat': str,
        'time': str,
        'flag': np.uint8,
        }

    # Load csv file
    tables = {}
    for tname, tschema in info['tables'].items():
      table_start_time = time.time()
      print('\tloading table %s' % tname, end='', flush=True)
      # Build attr2dtype map
      attr2dtype = {}
      n_emb = 0
      for attr, typ in tschema.items():
        if typ.startswith('emb'):
          n_emb += 1
          continue
        for typ_prefix, dtype in type2dtype.items():
          if typ.startswith(typ_prefix):
            assert attr not in attr2dtype
            attr2dtype[attr] = dtype
        assert attr in attr2dtype, '%s not in %s' % (attr, attr2dtype)
      assert tname in filenames, \
          'table %s is not found in filesystem %s.' % (tname, filenames)
      # Load pandas.dataframe
      tables[tname] = pd.read_csv(
          os.path.join(self.database_path, filenames[tname]),
          sep=info['sep'], dtype=attr2dtype, engine='c', memory_map=True)
      assert len(tschema) == len(tables[tname].columns) + n_emb, \
          '%s %s %d' % (tschema, tables[tname].columns, n_emb)
      print('\t%.2fs' % (time.time() - table_start_time))

    # Merge train & test tables
    if 'flag' not in info['tables'][main_table].values():
      # TODO: merge emb.npy
      assert '_flag' not in info['tables'][main_table].keys(), \
          '_flag has been in tschema of table %s' % main_table
      tables[main_table]['_flag'] = np.ones(
          tables[main_table].shape[0], dtype=np.uint8)
      tables[main_table + '_test']['_flag'] = np.zeros(
          tables[main_table + '_test'].shape[0], dtype=np.uint8)
      info['tables'][main_table]['_flag'] = 'flag'
      main_data = pd.concat(
          [tables[main_table], tables[main_table + '_test']],
          ignore_index=True)
      tables[main_table] = main_data
      del tables[main_table + '_test']
      del info['tables'][main_table + '_test']

    # Re-format time 
    for tname, tschema in info['tables'].items():
      table = tables[tname]
      for attr, typ in tschema.items():
        if typ.startswith('time'):
          time_format_type = typ.split(':', 1)[1] if ':' in typ else 'ms'
          if time_format_type in ['s', 'ms', 'us', 'ns']:
            table[attr] = pd.to_datetime(
                table[attr], unit=time_format_type)
          else:
            table[attr] = pd.to_datetime(
                table[attr], format=time_format_type)
    for tname, tschema in info['tables'].items():
      tschema_copy = copy.deepcopy(tschema)
      for attr, typ in tschema_copy.items():
        if typ.startswith('time'):
          tschema[attr] = 'time'

    # Add label type
    main_table, label_column = info['label'].split(':')
    info['tables'][main_table][label_column] = 'label'
    del info['label']

    # Transfer pandas.dataframe to our Table class
    self.tables = {}
    for tname, tdataframe in tables.items():
      self.tables[tname] = Table(self, tname, tdataframe)
    del tables
    self.main_table = self.tables[main_table]
    self.label = self.main_table[label_column]


  # ============================== 属性函数 ===================================

  @property
  def attr_columns(self):
    return list([c
      for t in sorted(self.tables.values()) for c in sorted(t.columns.values())
      if c.type.startswith('cat') or c.type.startswith('num')])

  @property
  def cat_columns(self):
    return list([c
      for t in sorted(self.tables.values()) for c in sorted(t.columns.values())
      if c.type.startswith('cat')])

  @property
  def num_columns(self):
    return list([c
      for t in sorted(self.tables.values()) for c in sorted(t.columns.values())
      if c.type.startswith('num')])

  @property
  def idcat_columns(self):
    return list([c
      for t in sorted(self.tables.values()) for c in sorted(t.columns.values())
      if c.type.startswith('id') or c.type.startswith('cat')])

  @property
  def time_columns(self):
    return list([c
      for t in sorted(self.tables.values()) for c in sorted(t.columns.values())
      if c.type.startswith('time')])

  @property
  def flag(self):
    flag = None
    for table in self.tables.values():
      for c in table.columns.values():
        if c.type == 'flag':
          assert flag is None
          flag = c
    return flag

  @property
  def data_memory_occupied(self):
    return sum([t.data_memory_occupied for t in self.tables.values()])

  @property
  def keyindex_memory_occupied(self):
    return sum([t.keyindex_memory_occupied for t in self.tables.values()])

  # ============================== 格式化输出函数 ================================

  def check_info(self):
    """ 验证 info 的有效性 """
    info, tables = self.info, self.tables

    # Check label
    label = None
    for tname, tschema in info['tables'].items():
      for attr, typ in tschema.items():
        if typ == 'label':
          assert label is None
          label = tname + ':' + attr
    assert label is not None

    # Check flag
    flag = None
    for tname, tschema in info['tables'].items():
      for attr, typ in tschema.items():
        if typ == 'flag':
          assert flag is None
          flag = tname + ':' + attr
    assert flag is None or flag.split(':')[0] == label.split(':')[0]

    # Check the consistance of attributes
    type_consistance = {
        'label':     [np.float32],
        'flag':      [np.uint8],
        'cat':       [np.int32],
        'id':        [np.int32],
        'num':       [np.float32],
        'time':      [np.dtype('<M8[ns]')],
        'emb':       [np.float32],
        'order':     [np.uint32],
    }
    for tname, tschema in sorted(info['tables'].items(), key=lambda x: x[0]):
      table = tables[tname]
      assert len(tschema) == len(table.columns), \
             '%s\n%s' % (tname, set(tschema).symmetric_difference(
             set(table.columns)))

      for col in table.columns.values():
        pd_type = col.data.dtype
        assert pd_type in type_consistance[tschema[col.name]], \
            "Incompatible type in json for %s:%s %s vs %s" % \
            (tname, col.name, type_consistance[tschema[col.name]], pd_type)

    # Check time_col
    if 'time_col' in info:
      time_col = info['time_col']
      for table in info['tables'].values():
        assert time_col not in table or table[time_col] == 'time'

    # Check block_manager
    for tname, tschema in info['tables'].items():
      for attr, typ in info['tables'].items():
        if typ in ['cat', 'id']:
          elem = tname + ':' + attr
          bid = self.block_manager.col2block[elem]
          assert elem in self.block_manager.blocks['columns']

  def save_data(self):
    # Save database to self.output_path
    if self.output_path is None:
      return
    os.makedirs(self.output_path, exist_ok=True)
    print('Save to', self.output_path)

    # Save info.json
    with open(os.path.join(self.output_path, 'info.json'), 'w') as f:
      json.dump(self.info, f, indent=4, sort_keys=True)
    # Save data.npy for every column
    data_path = os.path.join(self.output_path, 'data')
    os.makedirs(data_path, exist_ok=True)
    for tname, table in self.tables.items():
      os.makedirs(os.path.join(data_path, tname), exist_ok=True)
      for attr in table.columns:
        with open(os.path.join(data_path, tname, attr), 'wb') as f:
          np.save(f, table[attr].data)

  def stat(self, brief=False):
    def col_str(table):
      col_str = []
      if table.id is not None:
        col_str.append('1 id')
      if len(table.cat_columns) > 0:
        col_str.append('%d cat' % len(table.cat_columns))
      if len(table.num_columns) > 0:
        col_str.append('%d num' % len(table.num_columns))
      if len(table.time_columns) > 0:
        col_str.append('%d time' % len(table.time_columns))
      if len(table.emb_columns) > 0:
        col_str.append('%d emb' % len(table.emb_columns))
      if table.flag is not None:
        col_str.append('1 flag')
      if table.label is not None:
        col_str.append('1 label')
      if table.order is not None:
        col_str.append('ordered')
      return ', '.join(col_str)

    d_memory, d_metric = number_human_format(self.data_memory_occupied)
    k_memory, k_metric = number_human_format(self.keyindex_memory_occupied)
    title_str = '%s   %d tables   %d columns   %d lines   %.1f%s + %.1f%s' % (
          self.name, len(self.tables),
          sum([len(t.columns) for t in self.tables.values()]),
          sum([t.n_lines for t in self.tables.values()]),
          d_memory, d_metric, k_memory, k_metric)
    print('======================== %s ========================' % title_str)

    tb = pt.PrettyTable()
    tb.title = 'Dataset Statistics'
    tb.field_names = ['tables', 'lines', 'columns', 'memory']
    for table_name, table in sorted(self.tables.items(), key=lambda x:x[0]):
      d_memory, d_metric = number_human_format(table.data_memory_occupied)
      k_memory, k_metric = number_human_format(table.keyindex_memory_occupied)
      tb.add_row([table_name, table.n_lines, col_str(table),
                  '%.1f%s + %.1f%s' % (d_memory, d_metric, k_memory, k_metric) ])
    tb.float_format = '.2'
    print(tb)
    if brief:
      return

    for table_name, table in sorted(self.tables.items(), key=lambda x:x[0]):
      tb = pt.PrettyTable()
      d_memory, d_metric = number_human_format(table.data_memory_occupied)
      k_memory, k_metric = number_human_format(table.keyindex_memory_occupied)
      tb.title = '%s   %d lines   %s   %.1f%s + %.1f%s' % (
          table_name, table.n_lines, col_str(table), d_memory, d_metric, k_memory, k_metric)
      tb.field_names = ['name', 'type', 'missing', 'card', 'occupy', 'min', 'mean', 'max']

      if table.id is not None:
        tb.add_row([table.id.name, 'id', '0.00%', 'one', '100.00%',
                    1, '', table.id.block['unique']])

      def func(col):
        if hasattr(col, 'keyindex'):
          card = 'one' if col.keyindex.is_one else 'many'
          n_unique = ((col.keyindex.end[1:] - col.keyindex.start[1:]) != 0).sum()
        else:
          card = ''
          cat_unique = np.unique(col.data)
          n_unique = cat_unique.shape[0] if cat_unique[0] != 0 else cat_unique.shape[0]-1
        return card, n_unique, col.data.min(), col.data.max()

      columns = table.cat_columns
      rets = parallel(func, columns)
      for col, ret in zip(columns, rets):
        card, n_unique, data_min, data_max = ret
        tb.add_row([col.name, col.type,
                    '%.2f%%' % ((col.data == 0).mean() * 100), card,
                    '%.2f%%' % (n_unique / col.block['unique'] * 100),
                    data_min, '', data_max])

      def func(col):
        col_data_no_nan = col.data[~np.isnan(col.data)]
        if col_data_no_nan.shape[0] > 0:
          return (1 - col_data_no_nan.shape[0] / col.data.shape[0]) * 100, \
              float(np.min(col_data_no_nan)), float(np.mean(col_data_no_nan)), \
              float(np.max(col_data_no_nan))
        else:
          return 100, np.NaN, np.NaN, np.NaN

      columns = table.num_columns
      rets = parallel(func, columns)
      for col, ret in zip(columns, rets):
        missing, data_min, data_mean, data_max = ret
        tb.add_row([col.name, col.type,
                    '%.2f%%' % missing, '', '',
                    data_min, data_mean, data_max])

      for col in table.time_columns:
        tb.add_row([col.name, col.type,
                    '%.2f%%' % (pd.isnull(col.data).mean() * 100), '', '',
                    col.data.min(), '', col.data.max()])

      for col in table.emb_columns:
        tb.add_row([col.name, col.type + str(col.data.shape[1]),
                    '%.2f%%' % ((col.data == 0).mean() * 100), '', '',
                    col.data.min(), col.data.mean(), col.data.max()])

      if table.order is not None and col.type[0] != 't':
        col = table.order
        col_data = col.data[col.data != 0]
        tb.add_row([col.name, col.type,
                    '%.2f%%' % ((1 - col_data.shape[0] / col.data.shape[0]) * 100), '', '',
                    col_data.min(), '', col_data.max()]) # TODO: time mean value

      if table.flag is not None:
        col = table.flag
        col_data = col.data[~np.isnan(col.data)]
        tb.add_row([col.name, col.type,
                    '%.2f%%' % ((1 - col_data.shape[0] / col.data.shape[0]) * 100), '', '',
                    col_data.min(), col_data.mean(), col_data.max()])

      if table.label is not None:
        col = table.label
        col_data = col.data[~np.isnan(col.data)]
        tb.add_row([col.name, col.type,
                    '%.2f%%' % ((1 - col_data.shape[0] / col.data.shape[0]) * 100), '', '',
                    col_data.min(), col_data.mean(), col_data.max()])


      tb.float_format = '.2'
      print(tb)

  # ============================== 功能函数 ===================================

  def locate_col_by_gname(self, gname):
    tname, cname = gname.split(':', 1)
    return self.tables[tname].columns[cname]

  def fetch_main_data(self):
    """ 从主表中根据 flag 提取出训练集和测试集 """
    # TODO
    main_table = self.tables[self.main_table]
    flag = self.locateColumnByType(self.main_table, 'flag', is_unique=True)
    assert flag, 'Must have flag columns in table %s' % main_table
    train_data = main_table.loc[main_table[flag] == 1].drop(flag, axis=1)
    test_data = main_table.loc[main_table[flag] == 0].drop(flag, axis=1)
    return train_data, test_data

  def preprocess(self, tag):
    """ 预处理 """
    if len(tag) == 0 or tag.lower() == 'proper':
      FeatEngine(ProperFeatPipeline()).fit_transform(self)
    elif tag.lower() == 'basic':
      FeatEngine(BasicFeatPipeline()).fit_transform(self)
    elif tag.lower() == 'simple':
      FeatEngine(SimpleFeatPipeline()).fit_transform(self)
    elif tag.lower() == 'full':
      FeatEngine(FullFeatPipeline()).fit_transform(self)
    elif tag.lower() == 'complex':
      FeatEngine(ComplexFeatPipeline()).fit_transform(self)
    elif tag.lower() == 'test':
      FeatEngine(TestFeatPipeline()).fit_transform(self)
    else:
      assert False, 'Invalid prerocess tag.'
    self.check_info()

  def freeze(self):
    print(f'Freeze database {self.name}', end='')
    start_time = time.time()

    for table in sorted(self.tables.values()):
      if len(table.cat_columns) > 0:
        table.cat_feat = np.stack(
            [c.data for c in table.cat_columns],
            axis=1)
        for col in table.cat_columns:
          del col.data
      else:
        table.cat_feat = np.zeros((table.n_lines, 0), np.int32)
      assert table.cat_feat.dtype == np.int32, table.cat_feat.dtype

      if len(table.num_columns + table.emb_columns) > 0:
        table.num_feat = np.concatenate(
            [np.stack([c.data for c in table.num_columns], axis=1)] + \
            [c.data for c in table.emb_columns], axis=1)
        for col in table.num_columns + table.emb_columns:
          del col.data
      else:
        table.num_feat = np.zeros((table.n_lines, 0), np.float32)
      assert table.num_feat.dtype == np.float32

      table.is_freeze = True
      gc.collect()
    print(' cost %.1fs' % (time.time() - start_time))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--database_paths', type=str, default='', help='Database paths')
  parser.add_argument('--output_paths', type=str, default='', help='Output paths')
  parser.add_argument('--preprocess', type=str, default='', help='Preprocess tag')
  parser.add_argument('-detail', action='store_true', help='Whether stat is brief')
  args, _ = parser.parse_known_args()

  database_paths = args.database_paths.split(',')
  assert len(database_paths) != 0, 'database_paths is required.'
  output_paths = args.output_paths.split(',')
  assert len(output_paths) == 0 or len(output_paths) == len(database_paths), \
      'number of output_paths must be null or equal to number of database_paths.'

  for database_path, output_path in zip(database_paths, output_paths):
    database_path = os.path.expanduser(database_path)
    output_path = os.path.expanduser(output_path) if len(output_path) != 0 else None
    with Logger(output_path, git_log=False) as logger:
      database = Database(database_path, output_path)
      database.preprocess(args.preprocess)
      database.stat(brief=not args.detail)
      database.save_data()
      database.freeze()
  print('All finished.')
