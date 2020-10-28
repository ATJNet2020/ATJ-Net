import os
import json
import numpy as np

std_path, new_path = '../../backup/data', 'log'
# std_type = 'raw'
std_type = 'freeze'

for v in 'abcde':
  # Check freq
  std_freq = []
  for fname in os.listdir(os.path.join(std_path, v, 'freq')):
    fpath = os.path.join(std_path, v, 'freq', fname)
    with open(fpath) as f:
      std_freq.append((fpath, f.read()))
  std_freq.sort(key=lambda x: x[1])

  new_freq = []
  for fname in os.listdir(os.path.join(new_path, v, 'freq')):
    fpath = os.path.join(new_path, v, 'freq', fname)
    with open(fpath) as f:
      new_freq.append((fpath, f.read()))
  new_freq.sort(key=lambda x: x[1])

  for std, new in zip(std_freq, new_freq):
    assert std[1] == new[1], f'Diff: {std[0]} {std[0]}'

  # Check info
  with open(os.path.join(std_path, v, std_type, 'info.json')) as f:
    std_info = json.load(f)
  with open(os.path.join(new_path, v, 'info.json')) as f:
    new_info = json.load(f)
  assert set(std_info['tables'].keys()) == set(new_info['tables'].keys()), \
      f"{v} tname diff:\n{set(std_info['tables'].keys())}\n{set(new_info['tables'].keys())}"
  for tname in std_info['tables'].keys():
    assert len(std_info['tables'][tname]) == len(new_info['tables'][tname]), \
        f"{v}:{tname} len diff: {len(std_info['tables'][tname])} vs {len(new_info['tables'][tname])}"
    assert set(std_info['tables'][tname].keys()) == set(new_info['tables'][tname].keys()), \
        f"{v}:{tname} len diff:\n{set(std_info['tables'][tname].keys())}\n{set(new_info['tables'][tname].keys())}"
    for attr, typ in std_info['tables'][tname].items():
      assert typ == new_info['tables'][tname][attr], \
          f'{v}:{tname}:{attr} {typ} vs {new_info["tables"][tname][attr]}'
  std_blocks = [json.dumps(v, sort_keys=True) for v in std_info['blocks']]
  std_blocks.sort()
  new_blocks = [json.dumps(v, sort_keys=True) for v in new_info['blocks'].values()]
  new_blocks.sort()
  assert std_blocks == new_blocks, '%s\n%s' % (std_blocks, new_blocks)

  # Check data
  for tname, tschema in std_info['tables'].items():
    for attr, typ in tschema.items():
      if typ == 'id': continue
      std_fname = os.path.join(std_path, v, std_type, tname, attr)
      with open(std_fname, 'rb') as f:
        std_data = np.load(f)
      new_fname = os.path.join(new_path, v, 'raw', tname, attr)
      with open(new_fname, 'rb') as f:
        new_data = np.load(f)
      if typ == 'time':
        new_data[ np.isnat(new_data) ] = 0
        new_data = (new_data.astype(np.int64) / 1e9).astype(np.uint32)
      np.testing.assert_almost_equal(
          std_data, new_data, err_msg=f'{std_fname} {new_fname}', decimal=5)
