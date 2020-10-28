# -*- coding: future_fstrings -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import ctypes
import unittest
import numpy as np
from scipy import stats


class KeyIndex:
  def __init__(self, start, end, index, time=None):
    assert start.ndim == 1 and end.ndim == 1 and index.ndim == 1
    assert start.shape == end.shape
    assert end.shape[0] == 0 or index.shape[0] == end[-1], \
        f'{index.shape[0]}, {end[-1]}'
    assert start.dtype == np.int32 and end.dtype == np.int32
    assert index.dtype == np.int32
    assert time is None or (time.shape == index.shape and time.dtype == np.uint32)
    self.start = start
    self.end = end
    self.index = index
    is_one = np.all(np.less_equal(end - start, 1))
    self.is_one = is_one
    self.time = time

  @property
  def memory_occupied(self):
    return self.start.size * self.start.itemsize \
         + self.end.size * self.end.itemsize \
         + self.index.size * self.index.itemsize

lib = ctypes.cdll.LoadLibrary('./.index.so')


lib.build_key_index.argtypes = [
    ctypes.c_void_p, # data
    ctypes.c_int,    # n_data
    ctypes.c_void_p, # start
    ctypes.c_void_p, # end
    ctypes.c_int,    # n_idx
    ]

def build_key_index(col_data, order_data=None, n_idx=None, set_time=False):
  if order_data is None:
    index = np.argsort(col_data)
  else:
    index = np.lexsort([order_data, col_data])
  index = index.astype(np.int32, copy=False)

  data = col_data[index]
  data = data.astype(np.int32, copy=False)

  if n_idx is None:
    n_idx = data[-1] + 1
  else:
    assert data[-1] <= n_idx
    n_idx += 1
  start = np.empty(n_idx, np.int32)
  end   = np.empty(n_idx, np.int32)

  lib.build_key_index(
      data.ctypes.data_as(ctypes.c_void_p),
      data.shape[0],
      start.ctypes.data_as(ctypes.c_void_p),
      end.ctypes.data_as(ctypes.c_void_p),
      n_idx)
  if set_time:
    return KeyIndex(start, end, index, order_data[index])
  else:
    return KeyIndex(start, end, index)

class TestBuildKeyIndex(unittest.TestCase):

  def test_case(self):
    data = np.array([3, 5, 5, 4, 3, 3, 8])
    key_index = build_key_index(data)
    np.testing.assert_array_equal(
        key_index.index, [0, 4, 5, 3, 1, 2, 6])
    np.testing.assert_array_equal(
        key_index.start, [0, 0, 0, 0, 3, 4, 6, 6, 6])
    np.testing.assert_array_equal(
        key_index.end, [0, 0, 0, 3, 4, 6, 6, 6, 7])
    np.testing.assert_equal(
        key_index.is_one, False)

  def test_case_2(self):
    data = np.array([3, 5, 4, 1, 8])
    key_index = build_key_index(data)
    np.testing.assert_array_equal(
        key_index.index, [3, 0, 2, 1, 4])
    np.testing.assert_array_equal(
        key_index.start, [0, 0, 1, 1, 2, 3, 4, 4, 4])
    np.testing.assert_array_equal(
        key_index.end, [0, 1, 1, 2, 3, 4, 4, 4, 5])
    np.testing.assert_equal(
        key_index.is_one, True)



lib.fetch_key_index.argtypes = [
    ctypes.c_void_p, # data
    ctypes.c_void_p, # start
    ctypes.c_void_p, # end
    ctypes.c_int,    # n_idx
    ctypes.c_void_p, # res
    ctypes.c_int,    # n_res
    ctypes.c_int     # max_sample
]

def fetch_key_index(key_index, idx, max_sample=-1):
  assert idx.ndim == 1
  assert key_index.start.dtype == np.int32
  assert key_index.end.dtype == np.int32
  assert key_index.index.dtype == np.int32
  assert idx.dtype == np.int32
  if idx.shape[0] == 0:
    return KeyIndex(np.array([], np.int32),
        np.array([], np.int32), np.array([], np.int32))
  start = safe_index(key_index.start, idx)
  end   = safe_index(key_index.end,   idx)
  length = end - start
  if max_sample >= 0:
    length = np.minimum(length, max_sample)
  n_res = length.sum()
  res = np.empty(n_res, dtype=np.int32)
  lib.fetch_key_index(
      key_index.index.ctypes.data_as(ctypes.c_void_p),
      start.ctypes.data_as(ctypes.c_void_p),
      end.ctypes.data_as(ctypes.c_void_p),
      start.shape[0],
      res.ctypes.data_as(ctypes.c_void_p),
      n_res,
      max_sample)
  return KeyIndex(start, end, res)

class TestFetchKeyIndex(unittest.TestCase):

  def test_case(self):
    data = np.array([3, 5, 5, 4, 3, 3, 8], np.int32)
    idx = np.array([3, 2, 9, 5], np.int32)
    key_index = build_key_index(data)
    res_index = fetch_key_index(key_index, idx)
    np.testing.assert_array_equal(
        res_index.index, [0, 4, 5, 1, 2])
    np.testing.assert_array_equal(
        res_index.start, [0, 3, 3, 3])
    np.testing.assert_array_equal(
        res_index.end, [3, 3, 3, 5])



lib.safe_index_int.argtypes = [
    ctypes.c_void_p, # data
    ctypes.c_int,    # n_data
    ctypes.c_int,    # n_dim
    ctypes.c_void_p, # idx
    ctypes.c_int,    # n_idx
    ctypes.c_void_p, # res
    ]
lib.safe_index_float.argtypes = [
    ctypes.c_void_p, # data
    ctypes.c_int,    # n_data
    ctypes.c_int,    # n_dim
    ctypes.c_void_p, # idx
    ctypes.c_int,    # n_idx
    ctypes.c_void_p, # res
    ]

def safe_index(data, idx):
  """ 必须保证 data 的 memory 方式是 C-stype """
  assert idx.ndim == 1 and idx.dtype == np.int32
  assert data.dtype == np.float32 or data.dtype == np.int32
  res_shape = list(data.shape)
  res_shape[0] = idx.shape[0]
  res = np.empty(res_shape, data.dtype)
  if data.dtype == np.int32:
    safe_index_func = lib.safe_index_int
  else:
    safe_index_func = lib.safe_index_float
  safe_index_func(
      data.ctypes.data_as(ctypes.c_void_p),
      data.shape[0],
      np.prod(data.shape) // data.shape[0],
      idx.ctypes.data_as(ctypes.c_void_p),
      idx.shape[0],
      res.ctypes.data_as(ctypes.c_void_p))
  return res

class TestSafeIndex(unittest.TestCase):

  def test_case(self):
    df  = np.array([0, 1, 2, 3, 4, 5], np.float32)
    di  = np.array([0, 1, 2, 3, 4, 5], np.int32)
    idx = np.array([0, -1, 6, 5, -2, 0, 3], np.int32)
    rf  = safe_index(df, idx)
    ri  = safe_index(di, idx)
    # np.testing.assert_array_equal(
    #     rf, [0.0, np.nan, np.nan, 5.0, np.nan, 0.0, 3.0])
    np.testing.assert_array_equal(
        rf, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0])
    np.testing.assert_array_equal(
        ri, [0, 0, 0, 5, 0, 0, 3])

  def test_case_2(self):
    df  = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], np.float32)
    di  = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [5, 5]], np.int32)
    idx = np.array([0, -1, 6, 5, -2, 0, 3], np.int32)
    rf  = safe_index(df, idx)
    ri  = safe_index(di, idx)
    # np.testing.assert_array_equal(
    #     rf, [[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan],
    #          [5.0, 5.0], [np.nan, np.nan], [0.0, 0.0], [3.0, 3.0]])
    np.testing.assert_array_equal(
        rf, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
             [5.0, 5.0], [0.0, 0.0], [0.0, 0.0], [3.0, 3.0]])
    np.testing.assert_array_equal(
        ri, [[0, 1], [0, 0], [0, 0], [5, 5], [0, 0], [0, 1], [6, 7]])



lib.group_by.argtypes = [
    ctypes.c_void_p, # start
    ctypes.c_void_p, # end
    ctypes.c_int,    # n_idx
    ctypes.c_void_p, # index
    ctypes.c_void_p, # target
    ctypes.c_void_p, # mean
    ctypes.c_void_p, # std
    ctypes.c_void_p, # skew
    ctypes.c_void_p, # kurt
    ]

def moment_group_by(key_index, target):
  assert key_index.index.shape == target.shape and target.dtype == np.float32
  assert key_index.start.shape == key_index.end.shape
  n_idx = key_index.start.shape[0]
  mean = np.empty(n_idx, np.float32)
  std = np.empty(n_idx, np.float32)
  skew = np.empty(n_idx, np.float32)
  kurt = np.empty(n_idx, np.float32)
  lib.group_by(
      key_index.start.ctypes.data_as(ctypes.c_void_p),
      key_index.end.ctypes.data_as(ctypes.c_void_p),
      n_idx,
      key_index.index.ctypes.data_as(ctypes.c_void_p),
      target.ctypes.data_as(ctypes.c_void_p),
      mean.ctypes.data_as(ctypes.c_void_p),
      std.ctypes.data_as(ctypes.c_void_p),
      skew.ctypes.data_as(ctypes.c_void_p),
      kurt.ctypes.data_as(ctypes.c_void_p))
  return mean, std, skew, kurt

class TestMomentGroupBy(unittest.TestCase):

  def test_case(self):
    length = np.random.randint(0, 10, 20, np.int32)
    end = np.cumsum(length).astype(np.int32)
    start = np.concatenate([[0], end[:-1]]).astype(np.int32)
    index = np.arange(end[-1]).astype(np.int32)
    np.random.shuffle(index)
    key_index = KeyIndex(start, end, index)
    target = np.random.rand(end[-1]).astype(np.float32)
    target[np.random.randint(0, end[-1], 10)] = np.nan
    mean, std, skew, kurt = moment_group_by(key_index, target)
    for i in range(len(start)):
      x = target[index[start[i] : end[i]]]
      x = x[~np.isnan(x)]
      if len(x) == 0:
        self.assertTrue(np.isnan(mean[i]))
        self.assertTrue(np.isnan(std[i]))
        self.assertTrue(np.isnan(skew[i]))
        self.assertTrue(np.isnan(kurt[i]))
      else:
        self.assertTrue(np.all(np.isclose(np.mean(x), mean[i], atol=1e-4, equal_nan=True)))
        self.assertTrue(np.all(np.isclose(np.std(x), std[i], atol=1e-4, equal_nan=True)))
        self.assertTrue(np.all(np.isclose(stats.skew(x), skew[i], atol=1e-4, equal_nan=True)))
        self.assertTrue(np.all(np.isclose(stats.kurtosis(x), kurt[i], atol=1e-4, equal_nan=True)))


lib.arange_keyindex.argtypes = [
    ctypes.c_void_p, # start
    ctypes.c_void_p, # end
    ctypes.c_void_p, # arr
    ctypes.c_int,    # m
    ctypes.c_int,    # n
    ]

def arange_keyindex(key_index):
  size = key_index.end - key_index.start
  m = size.shape[0]
  n = size.max() if m > 0 else 0
  arr = np.empty((m, n), dtype=np.int32, order='C')
  lib.arange_keyindex(
      key_index.start.ctypes.data_as(ctypes.c_void_p),
      key_index.end.ctypes.data_as(ctypes.c_void_p),
      arr.ctypes.data_as(ctypes.c_void_p),
      m, n)
  return arr

class TestArangeKeyIndex(unittest.TestCase):

  def test_case(self):
    key_index = KeyIndex(
        np.array([0, 3, 3, 10, 10], np.int32),
        np.array([3, 3, 10, 10, 11], np.int32),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.int32))
    idx = arange_keyindex(key_index)
    np.testing.assert_array_equal(idx, [
        [ 0,  1,  2, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [ 3,  4,  5,  6,  7,  8,  9],
        [-1, -1, -1, -1, -1, -1, -1],
        [10, -1, -1, -1, -1, -1, -1]])


if __name__ == '__main__':
  unittest.main()
