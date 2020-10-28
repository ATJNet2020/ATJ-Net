import os
import sys
import CONSTANT
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import Pool


def parallel(func, candicates):
  if len(candicates) < 8:
    return [ func(cand) for cand in candicates ]
  else:
    return Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(
        delayed(func)(cand) for cand in candicates)

def parallel_map(func, candicates):
  if len(candicates) < 8:
    return [ func(cand) for cand in candicates ]
  else:
    pool = Pool(processes=CONSTANT.JOBS)
    result = pool.map(func, candicates)
    pool.close()
    pool.join()
    return result

def parallel_loop(func, candicates):
  return [ func(cand) for cand in candicates ]


def is_nan_column(typ, data):
  assert isinstance(data, np.ndarray)
  if typ == 'cat':
    return np.mean(data == 0) > 0.8
  elif typ == 'num':
    return np.mean(np.isnan(data)) > 0.8
  else:
    assert False

def is_dup_column(typ, data):
  sample_limit = 100000
  random_state = np.random.RandomState(1021)
  assert isinstance(data, np.ndarray)
  if data.shape[0] <= sample_limit:
    sample_data = data
  else:
    sample_data = random_state.choice(data, sample_limit)

  if typ == 'num':
    no_nan_data = sample_data[~np.isnan(sample_data)]
  else:
    no_nan_data = sample_data[sample_data != 0]
  unique_data, unique_count = np.unique(no_nan_data, return_counts=True)
  return len(unique_data) <= 1 or \
         (typ == 'num' and unique_data[-1] - unique_data[0] < 0.01) \
         or unique_count.max() > sample_limit * (1 - 1e-4)


def number_human_format(num):
  if num < 1024:
    return num, 'B'
  elif num < (1024**2):
    return num/1024, 'K'
  elif num < (1024**3):
    return num/(1024**2), 'M'
  elif num < (1024**4):
    return num/(1024**3), 'G'
  else:
    return num/(1024**4), 'T'

def time_human_format(interval):
  if interval < 60:
    return interval, 's'
  elif interval < 60*60:
    return interval/60, 'm'
  elif interval < 60*60*24:
    return interval/60/60, 'h'
  else:
    return interval/60/60/24, 'd'


class Logger(object):
  def __init__(self, output_dir, git_log=False):
    self.output_dir = output_dir
    self.git_log = git_log

  def __enter__(self):
    if self.output_dir is None or len(self.output_dir) == 0: return
    self.terminal = sys.stdout
    os.makedirs(self.output_dir, exist_ok=True)
    self.log_path = os.path.join(self.output_dir, 'log.txt')
    self.log = open(self.log_path, "w")
    if self.git_log:
      gitLogDiff(os.path.join(self.output_dir, 'git.txt'))
    sys.stdout = self

  def __exit__(self, type, value, trace):
    if self.output_dir is None: return
    sys.stdout = self.terminal
    self.log.close()

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
    self.log.flush()

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def gitLogDiff(log_path):
  os.system('git log >"%s"' % log_path)
  os.system('git diff >>"%s"' % log_path)
