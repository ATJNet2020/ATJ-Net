import os
import json
import numpy as np
np.random.seed(1231)

categories = [
    'Beauty & Spas',
    'Health & Medical',
    'Home Services',
    'Restaurants',
    'Shopping',
]
yelp_dir = '../data/y/'
with open(yelp_dir + 'info.json') as f:
  info = json.load(f)
  info['label'] = 'review:label'
  del info['tables']['review']['useful']
  del info['tables']['review']['cool']
  del info['tables']['review']['funny']
  info['tables']['review']['stars'] = 'num'
  info['tables']['review']['label'] = 'num'
  info['tables']['review']['flag'] = 'flag'

for k, cat_name in enumerate(categories, 1):
  print('Building sub-dataset', cat_name)
  target_dir = yelp_dir.rstrip('/') + str(k) + '/'
  os.makedirs(target_dir, exist_ok=True)
  cat_name = cat_name.lower()

  with open(target_dir + '/info.json', 'w') as f:
    json.dump(info, f, indent=4)

  bids = set()
  with open(yelp_dir + 'business.data', 'r') as fin, \
      open(target_dir + 'business.data', 'w') as fout:
    head = fin.readline()
    fout.write(head)
    idx = head.strip().split('\t').index('categories')
    for line in fin:
      cols = line.strip().split('\t')
      bids.add(cols[0])
      if cat_name in cols[idx].lower():
        fout.write(line)

  uids = set()
  with open(yelp_dir + 'review.data', 'r') as fin, \
      open(target_dir + 'review.data', 'w') as fout:
    head = '\t'.join(fin.readline().strip().split('\t')[:-3]) + '\tlabel\tflag\n'
    fout.write(head)
    candidate_lines = []
    candidate_idx = []
    for i, line in enumerate(fin):
      cols = line.strip().split('\t')
      uids.add(cols[1])
      last_sum = sum(map(float, cols[-3:]))
      if cols[2] in bids and last_sum > 0:
        useful = float(cols[-3]) / last_sum
        label = '1' if useful >= 0.75 else '0'
        line = '\t'.join(cols[:-3]) + '\t' + label
        candidate_lines.append(line)
        candidate_idx.append(i)
    negs = np.random.choice(len(candidate_lines), round(len(candidate_lines)*0.2), False)
    flag = np.ones(len(candidate_lines), dtype=np.int64)
    flag[negs] = 0
    for i, line in enumerate(candidate_lines):
      line += '\t' + str(flag[i]) + '\n'
      fout.write(line)
    emb = np.load(yelp_dir + 'review_text.emb')[candidate_idx]
    with open(target_dir + 'review_text.emb', 'wb') as f:
      np.save(f, emb)

  with open(yelp_dir + 'user.data', 'r') as fin, \
      open(target_dir + 'user.data', 'w') as fout:
    head = fin.readline()
    fout.write(head)
    idx = head.strip().split('\t').index('friends')
    for line in fin:
      cols = line.strip().split('\t')
      if cols[0] in uids:
        candidate_friends = cols[idx].split(',')
        cols[idx] = ','.join([c for c in candidate_friends if c in uids])
        line = '\t'.join(cols) + '\n'
        fout.write(line)

  with open(yelp_dir + 'tip.data', 'r') as fin, \
      open(target_dir + 'tip.data', 'w') as fout:
    head = fin.readline()
    fout.write(head)
    candidate_idx = []
    for i, line in enumerate(fin):
      cols = line.strip().split('\t')
      if cols[-2] in bids and cols[-1] in uids:
        fout.write(line)
        candidate_idx.append(i)
    emb = np.load(yelp_dir + 'tip_text.emb')[candidate_idx]
    with open(target_dir + 'tip_text.emb', 'wb') as f:
      np.save(f, emb)
