# -*- coding: future_fstrings -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import os
import time
import numpy as np


class Column:

  def __init__(self, table, attr, data):
    self.table = table
    self.name = attr
    self.data = data

  def __lt__(self, other):
    return self.name < other.name

  @property
  def type(self):
    return self.table.info[self.name]

  @property
  def global_name(self):
    return self.table.name + ':' + self.name

  @property
  def n_lines(self):
    if self.table.is_freeze:
      return self.table.cat_feat.shape[0]
    else:
      return self.data.shape[0]

  @property
  def block(self):
    block_manager = self.table.database.block_manager
    gname = self.global_name
    block = block_manager.blocks[block_manager.col2block[gname]]
    assert gname in block['columns']
    return block

  @property
  def block_id(self):
    return self.table.database.block_manager.col2block[self.global_name]

  @property
  def data_memory_occupied(self):
    if self.table.is_freeze:
      return self.table.cat_feat.shape[0] * self.table.cat_feat.itemsize
    else:
      return self.data.size * self.data.itemsize

  @property
  def keyindex_memory_occupied(self):
    if not hasattr(self, 'keyindex'):
      return 0
    return self.keyindex.start.size * self.keyindex.start.itemsize \
         + self.keyindex.end.size * self.keyindex.end.itemsize \
         + self.keyindex.index.size * self.keyindex.index.itemsize


class Table:

  def __init__(self, database, tname, tdataframe):
    self.database = database
    self.name = tname
    self.info = database.info['tables'][tname]
    self.columns = {}
    for attr in list(sorted(self.info.keys())):
      if self.info[attr].startswith('emb'):
        emb_path = os.path.join(database.database_path, self.info[attr].split(':', 1)[1])
        self.columns[attr] = Column(self, attr, np.load(emb_path))
        self.info[attr] = 'emb'
      else:
        self.columns[attr] = Column(self, attr, tdataframe[attr].to_numpy())
    self.is_freeze = False
    self.n_lines = tdataframe.shape[0]
    for col in self.columns.values():
      assert self.n_lines == col.n_lines

  def __lt__(self, other):
    return self.name < other.name

  def __getitem__(self, attr):
    return self.columns[attr]

  def __setitem__(self, attr, val):
    self.columns[attr] = val

  def __delitem__(self, attr):
    del self.columns[attr]

  @property
  def attr_columns(self):
    return list(sorted([c for c in self.columns.values()
      if c.type.startswith('cat') or c.type.startswith('num')]))

  @property
  def cat_columns(self):
    return list(sorted([c for c in self.columns.values()
      if c.type.startswith('cat')]))

  @property
  def num_columns(self):
    return list(sorted([c for c in self.columns.values()
      if c.type.startswith('num')]))

  @property
  def idcat_columns(self):
    return list(sorted([c for c in self.columns.values()
      if c.type.startswith('id') or c.type.startswith('cat')]))

  @property
  def multicat_columns(self):
    return list(sorted([c for c in self.columns.values()
      if c.type.startswith('multi-cat')]))

  @property
  def emb_columns(self):
    return list(sorted([c for c in self.columns.values()
      if c.type.startswith('emb')]))

  @property
  def time_columns(self):
    return list(sorted([c for c in self.columns.values()
      if c.type.startswith('time')]))

  @property
  def order(self):
    order = None
    for c in self.columns.values():
      if c.type == 'order':
        assert order is None
        order = c
      if c.type == 'time' and \
          'time_col' in self.database.info and \
          c.name == self.database.info['time_col']:
        assert order is None
        order = c
    return order

  @property
  def id(self):
    id = None
    for c in self.columns.values():
      if c.type == 'id':
        assert id is None
        id = c
    return id

  @property
  def flag(self):
    label = None
    for c in self.columns.values():
      if c.type == 'flag':
        assert label is None
        label = c
    return label

  @property
  def label(self):
    label = None
    for c in self.columns.values():
      if c.type == 'label':
        assert label is None
        label = c
    return label
  
  @property
  def data_memory_occupied(self):
    return sum([col.data_memory_occupied for col in self.columns.values()])

  @property
  def keyindex_memory_occupied(self):
    return sum([col.keyindex_memory_occupied for col in self.columns.values()])

  def add_column(self, name, typ, data, ignore_block_manager=False):
    assert typ in ['id', 'cat', 'num'] # TODO: add cat
    assert name not in self.columns and name not in self.info, name
    assert data.ndim == 1 and data.shape[0] == self.n_lines
    self.info[name] = typ
    self.columns[name] = Column(self, name, data)
    if typ in ['id', 'cat'] and not ignore_block_manager:
      self.database.block_manager.add_column(self.name, name)
    return self.columns[name]

  def drop_column(self, name, ignore_block_manager=False):
    assert name in self.columns and name in self.info, name
    col = self.columns[name]
    if col.type in ['id', 'cat'] and not ignore_block_manager:
      self.database.block_manager.drop_column(self.name, name)
    del self.columns[name]
    del self.info[name]
    return col
