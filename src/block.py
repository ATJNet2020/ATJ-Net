import os
import copy
import time
from utils import parallel_map, parallel_loop
from encoder import Encoder_Py, Encoder_Cpp


# 全局变量是为了调用 multiprocessing.Pool
_global_block_manager = None
_global_database = None
_global_encoders = None
_global_encoder_cls = None

# 根据 block 和 col.data 构建 encoders
def _build_encoder(i):
  # counter 维护了合法值的频数
  encoder = _global_encoder_cls()
  for gname in _global_block_manager.blocks[i]['columns']:
    tname, cname = gname.split(':')
    data = _global_database.tables[tname][cname].data
    encoder.update(gname, data)

  # vec 维护了 (原始值,频数) 的数组，按照 key 函数排序
  encoder.vec()

  # 如果合法值小于 1，则该 block 无信息增益，可去掉
  if encoder.unique() <= 1:
    return i, None

  # 保存 vec 数组到 freq 目录
  if _global_database.output_path is not None:
    freq_path = os.path.join(_global_database.output_path, 'freq')
    os.makedirs(freq_path, exist_ok=True)
    freq_file = os.path.join(freq_path, str(i))
    encoder.save(freq_file)

  # 编码器维护了原始值到新值的映射
  encoder.encode()
  return i, encoder

# 根据 encoder 重新映射 col.data
def _remap_encoder(gname):
  encoder = _global_encoders[_global_block_manager.col2block[gname]]
  col = _global_database.locate_col_by_gname(gname)
  return encoder.remap(gname, col.data)

def _remap_template(parallel_1, parallel_2):
  start = time.time()
  # 并行计算每个 block 的编码器
  encoders = parallel_1(_build_encoder, _global_block_manager.blocks.keys())

  # 删除无效的 block 以及相应的列, 返回有效的编码器
  for bid, encoder in encoders:
    if encoder is None:
      _global_block_manager.blocks[bid]['unique'] = 0
      for elem in copy.deepcopy(_global_block_manager.blocks[bid]['columns']):
        tname, attr = elem.split(':')
        _global_database.tables[tname].drop_column(attr)
    else:
      _global_block_manager.blocks[bid]['unique'] = encoder.unique()
      _global_encoders[bid] = encoder

  print('\t\tBuild Encoders: %.2f' % (time.time() - start))
  start = time.time()

  # 重新映射 col.data
  todo_list = [col.global_name for col in _global_database.idcat_columns]
  results = parallel_2(_remap_encoder, todo_list)
  for col, ret in zip(_global_database.idcat_columns, results):
    col.data = ret

  print('\t\tRemap Cate: %.2f' % (time.time() - start))



class BlockManager:
  """ 帮助 Database 对象管理列之间的 joinable 关系
      相互之间可以 join 的列称为 block
  """
  def __init__(self, database):
    """ 直接从 database.info 构造 blocks """
    self.database = database
    info = self.database.info

    # col2block 维护了 '表名:列名' 到 block_id 的映射
    self.col2block = {}
    # n_blocks 仅供内部使用，用于记录最大创建的 block
    self.n_blocks = 0

    # Parse from relations
    for rel in info['relations']:
      bid = None
      for elem in rel:
        tname, attr = elem.split(':')
        assert tname in info['tables'] and attr in info['tables'][tname], \
          '%s not in table schema' % elem
        assert info['tables'][tname][attr] in ['id', 'cat'] or \
            info['tables'][tname][attr].startswith('multi-cat'), \
            'Relation of %s must be cat or multi-cat' % elem
        if elem in self.col2block:
          assert bid is None or bid == self.col2block[elem]
          bid = self.col2block[elem]
      if bid is None:
        bid = self.n_blocks
        self.n_blocks += 1
      for elem in rel:
        self.col2block[elem] = bid

    # Parse from single attribute
    for tname, tschema in sorted(info['tables'].items(), key=lambda x: x[0]):
      for attr, typ in sorted(tschema.items(), key=lambda x: x[0]):
        col = tname + ':' + attr
        if (typ in ['id', 'cat'] or typ.startswith('multi-cat')) \
            and col not in self.col2block:
          self.col2block[col] = self.n_blocks
          self.n_blocks += 1

    # blocks 维护了 block_id 到 block 的影射
    # block 包含两个域
    #     columns 描述了当前 block 包含的所有 '表名:列名'
    #     unique  描述了当前 block 可以取的独立值的个数 (不必须，不及时更新)
    self.blocks = {i : {'columns': []} for i in range(self.n_blocks)}
    for col, bid in self.col2block.items():
      self.blocks[bid]['columns'].append(col)
    for block in self.blocks.values():
      block['columns'].sort()
    self.database.info['blocks'] = self.blocks

  def add_column(self, tname, attr):
    """ 向管理器添加一列 """
    elem = tname + ':' + attr
    assert elem not in self.col2block
    assert self.database.info['tables'][tname][attr] in ['cat', 'id']
    self.col2block[elem] = self.n_blocks
    self.blocks[self.n_blocks] = {'columns': [elem]}
    self.n_blocks += 1
    return self.col2block[elem]

  def drop_column(self, tname, attr):
    """ 在管理器中删除一列 """
    elem = tname + ':' + attr
    bid = self.col2block[elem]
    del self.col2block[elem]
    self.blocks[bid]['columns'].remove(elem)
    if len(self.blocks[bid]['columns']) == 0:
      del self.blocks[bid]

  def reset_freq(self):
    """ 更新 block 的 unique 域
        更新每个 block 的所有可取值域到 freq 目录
        返回每个 block 的编码器，编码器维护了原始值到新值的影射
    """
    global _global_block_manager
    global _global_database
    global _global_encoders
    global _global_encoder_cls

    _global_block_manager = self
    _global_database = self.database
    _global_encoders = {}

    _global_encoder_cls = Encoder_Py
    _remap_template(parallel_map, parallel_map)
    # _global_encoder_cls = Encoder_Cpp
    # _remap_template(parallel_loop, parallel_loop)

    _global_block_manager = None
    _global_database = None
    _global_encoders = None
    _global_encoder_cls = None
