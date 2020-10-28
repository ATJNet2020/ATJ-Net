import json
import ctypes
import numpy as np
from collections import Counter


lib = ctypes.cdll.LoadLibrary('./.encoder.so')

lib.encoder_new.restype = ctypes.c_void_p
lib.encoder_destroy.argtypes = [ ctypes.c_void_p ]
lib.encoder_update.argtypes = [
    ctypes.c_void_p, # ptr
    ctypes.c_char_p, # name
    ctypes.c_void_p, # data
    ctypes.c_int,    # n
    ]
lib.encoder_vectorize.argtypes = [ ctypes.c_void_p ]
lib.encoder_size.argtypes = [ ctypes.c_void_p ]
lib.encoder_size.restype = ctypes.c_int
lib.encoder_save.argtypes = [
    ctypes.c_void_p, # ptr
    ctypes.c_char_p, # path
    ]
lib.encoder_encode.argtypes = [ ctypes.c_void_p ]
lib.encoder_remap.argtypes = [
    ctypes.c_void_p, # ptr
    ctypes.c_char_p, # name
    ctypes.c_void_p, # new_data
    ctypes.c_int,    # n
    ]

class Encoder_Cpp:

  def __init__(self):
    self.c_ptr = lib.encoder_new()

  def __del__(self):
    lib.encoder_destroy(self.c_ptr)

  def update(self, gname, data):
    data = data.astype(object)
    lib.encoder_update(
        self.c_ptr,
        gname.encode('utf-8'),
        data.ctypes.data_as(ctypes.c_void_p),
        data.shape[0])

  def vec(self):
    lib.encoder_vectorize(self.c_ptr)

  def unique(self):
    return lib.encoder_size(self.c_ptr)

  def save(self, path):
    lib.encoder_save(self.c_ptr, path.encode('utf-8'))

  def encode(self):
    lib.encoder_encode(self.c_ptr)

  def remap(self, name, data):
    n = data.shape[0]
    new_data = np.empty(n, dtype=np.int32)
    lib.encoder_remap(
        self.c_ptr,
        name.encode('utf-8'),
        new_data.ctypes.data_as(ctypes.c_void_p),
        n)
    return new_data


# 定义了原始值到新值的影射顺序
def _map_key(x):
  # try:
  #   return (-x[1], 0, int(x[0]))
  # except ValueError:
  #   return (-x[1], 1, x[0])
  return (-x[1], str(x[0]))

class Encoder_Py:

  def __init__(self):
    # counter 维护了合法值的频数
    self.counter = Counter()
    self.names = []

  # 更新列到 counter
  def update(self, gname, data):
    self.names.append(gname)
    self.counter.update(data)

  # vec 维护了 (原始值,频数) 的数组，按照 key 函数排序
  def vec(self):
    del self.counter[np.NAN]
    vec = self.counter.most_common()
    self.vec = list(sorted(vec, key=_map_key))
    self.names.sort()

  # unique 数目
  def unique(self):
    return len(self.vec)

  # 保存 vec 数组到 path 路径
  def save(self, path):
    info = {"columns": self.names, "unique": len(self.vec)}
    with open(path, 'w') as f:
      f.write(json.dumps(info, sort_keys=True) + '\n\n')
      for v in self.vec:
        f.write(str(v[0]) + '\t' + str(v[1]) + '\n')

  # 编码器维护了原始值到新值的映射
  def encode(self):
    self.encoder = {v[0]: i for v, i in zip(self.vec, range(1, 1+len(self.vec)))}

  # 根据 encoder 重新映射 col.data
  def remap(self, name, data):
    # 新值从 1 开始计数; 0 表示空值，用 np.NaN 代替
    def foo(x):
      if isinstance(x, float) and np.isnan(x): return 0
      else: return self.encoder[x]
    return np.vectorize(foo, otypes=[np.int32])(data)
