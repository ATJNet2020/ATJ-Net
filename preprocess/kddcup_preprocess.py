import os
import json
import copy
import shutil
import subprocess
import numpy as np
import pandas as pd

for v in 'ABCDE':
  random_state = np.random.RandomState(1021)
  raw_dir = os.path.join('../raw/kddcup', v, 'train')
  new_dir = os.path.join('../data', v.lower())
  if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
  os.makedirs(new_dir)

  print('transfer', v)

  with open(os.path.join(raw_dir, 'info.json')) as f:
    info = json.load(f)

  info['tables']['main']['label'] = 'num'
  info['label'] = 'main:label'
  info['tables']['main']['flag'] = 'flag'
  del info['time_budget']
  if 'start_time' in info: 
    del info['start_time']

  for tname, tschema in info['tables'].items():
    tschema_copy = copy.deepcopy(tschema)
    for attr, typ in tschema_copy.items():
      if typ == 'time':
        tschema[attr] = 'time:ms'

  new_rel = []
  for rel in info['relations']:
    new_rel.append([
      rel['table_A'] + ':' + rel['key'][0],
      rel['table_B'] + ':' + rel['key'][0] ])
  info['relations'] = new_rel

  with open(os.path.join(new_dir, 'info.json'), 'w') as f:
    json.dump(info, f, indent=4)

  file_path = os.path.join(raw_dir, 'main_train.data')
  n_line = int(subprocess.getoutput('wc -l %s' % (file_path)).split()[0]) - 1
  test_idx = random_state.choice(np.arange(n_line, dtype=np.int32), int(n_line*0.2))
  flag = np.ones(n_line, dtype=np.int32)
  flag[test_idx] = 0

  with open(os.path.join(raw_dir, 'main_train.data')) as fin_data, \
       open(os.path.join(raw_dir, 'main_train.solution')) as fin_solution, \
       open(os.path.join(new_dir, 'main.data'), 'w') as fout:
    for i, line in enumerate(fin_data):
      line = line.strip() + '\t' + fin_solution.readline().strip()
      if i == 0:
        line += '\tflag\n'
      else:
        line += f'\t{flag[i-1]}\n'
      fout.write(line)

  for f in os.listdir(raw_dir):
    if not f.startswith('main_train') and not f.startswith('info'):
      shutil.copy(os.path.join(raw_dir, f), os.path.join(new_dir, f))
