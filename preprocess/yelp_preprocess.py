import os
import time
import json
import string
import subprocess
from tqdm import tqdm
from collections import defaultdict

def getFileLines(filePath):
  return int(subprocess.getoutput('wc -l ' + filePath).split()[0])

fileDir = '../raw/yelp'
output_dir = '../data/y'
os.makedirs(output_dir, exist_ok=True)


def conv_review(info):
  print('Converting review:')
  fileName = 'review.json'
  with open(os.path.join(fileDir, fileName)) as fin, \
       open(os.path.join(output_dir, 'review.data'), 'w') as fout:
    col_names = ['review_id', 'user_id', 'business_id', 'stars',
                 'date', 'text', 'useful', 'funny', 'cool']
    col_types = ['id', 'cat', 'cat', 'label',
                 'time', 'text', 'num', 'num', 'num']
    info['tables']['review'] = dict(zip(col_names, col_types))
    print('\t'.join(col_names), file=fout)

    for line in tqdm(fin, total=getFileLines(os.path.join(fileDir, fileName))):
      line = json.loads(line)
      line['date'] = int(time.mktime(time.strptime(line['date'], '%Y-%m-%d %H:%M:%S'))) * 1000
      line['text'] = line['text'].replace('\t', ' ')
      line['text'] = line['text'].replace('\n', ' ')
      line['text'] = line['text'].replace('\r', '')
      print('\t'.join([str(line[c]) for c in col_names]), file=fout)


def conv_business(info):
  print('Converting business:')
  fileName = 'business.json'
  def func(attr, prefix, ret):
    if attr is None: return
    for k, v in attr.items():
      k = k.lower().replace('-', '_')
      if isinstance(v, dict):
        func(v, prefix + '_' + k, ret)
      elif isinstance(v, str):
        try:
          v = eval(v)
          if isinstance(v, dict):
            func(v, prefix + '_' + k, ret)
          else:
            ret[prefix + '_' + k].add(v)
        except:
          ret[prefix + '_' + k].add(v)
      else:
        ret[prefix + '_' + k].add(v)

  ret = defaultdict(set)
  with open(os.path.join(fileDir, fileName)) as fin:
    for line in fin:
      line = json.loads(line)
      func(line['attributes'], 'attr', ret)
    attrs = list(sorted(ret.keys()))

  with open(os.path.join(fileDir, fileName)) as fin, \
       open(os.path.join(output_dir, 'business.data'), 'w') as fout:
    col_names = [
        'business_id', 'name', 'address', 'city', 'state', 'postal_code',
        'latitude', 'longitude', 'stars', 'review_count', 'is_open',
        'categories'] + attrs
    col_types = [
        'id', 'cat', 'cat', 'cat', 'cat', 'cat',
        'num', 'num', 'num', 'num', 'num',
        'multi-cat'] + ['cat'] * len(attrs)
    info['tables']['business'] = dict(zip(col_names, col_types))
    print('\t'.join(col_names), file=fout)

    for line in tqdm(fin, total=getFileLines(os.path.join(fileDir, fileName))):
      line = json.loads(line)
      if line['categories'] is not None:
        line['categories'] = ','.join([
          v.strip() for v in line['categories'].split(',')])
      ret = defaultdict(set)
      func(line['attributes'], 'Attr', ret)
      for k, v in ret.items():
        assert len(v) == 1
        line[k] = v
      for attr in attrs:
        if attr not in line:
          line[attr] = ''
      print('\t'.join([str(line[c]) for c in col_names]), file=fout)


def conv_user(info):
  print('Converting user:')
  fileName = 'user.json'
  with open(os.path.join(fileDir, fileName)) as fin, \
       open(os.path.join(output_dir, 'user.data'), 'w') as fout:
    col_names = [
        'user_id', 'name', 'review_count', 'yelping_since', 'friends',
        'useful', 'funny', 'cool', 'fans', 'elite', 'average_stars',
        'compliment_hot', 'compliment_more', 'compliment_profile',
        'compliment_cute', 'compliment_list', 'compliment_note',
        'compliment_plain', 'compliment_cool', 'compliment_funny',
        'compliment_writer', 'compliment_photos']
    col_types = [
        'id', 'cat', 'num', 'time', 'multi-cat',
        'num', 'num', 'num', 'num', 'multi-cat', 'num',
        'num', 'num', 'num', 'num', 'num', 'num',
        'num', 'num', 'num', 'num', 'num']
    info['tables']['user'] = dict(zip(col_names, col_types))
    print('\t'.join(col_names), file=fout)

    for line in tqdm(fin, total=getFileLines(os.path.join(fileDir, fileName))):
      line = json.loads(line)
      line['yelping_since'] = int(time.mktime(time.strptime(line['yelping_since'],
          '%Y-%m-%d %H:%M:%S'))) * 1000
      line['friends'] = ','.join(line['friends'].split(', '))
      line['elite'] = ','.join(list(map(str, line['elite'].split(','))))
      print('\t'.join([str(line[c]) for c in col_names]), file=fout)


def conv_tip(info):
  print('Converting tip:')
  fileName = 'tip.json'
  with open(os.path.join(fileDir, fileName)) as fin, \
       open(os.path.join(output_dir, 'tip.data'), 'w') as fout:
    col_names = ['text', 'date', 'compliment_count', 'business_id', 'user_id']
    col_types = ['text', 'time', 'num', 'cat', 'cat']
    info['tables']['tip'] = dict(zip(col_names, col_types))
    print('\t'.join(col_names), file=fout)

    for line in tqdm(fin, total=getFileLines(os.path.join(fileDir, fileName))):
      line = json.loads(line)
      line['text'] = line['text'].replace('\t', ' ')
      line['text'] = line['text'].replace('\n', ' ')
      line['text'] = line['text'].replace('\r', '')
      line['date'] = int(time.mktime(time.strptime(line['date'], '%Y-%m-%d %H:%M:%S'))) * 1000
      print('\t'.join([str(line[c]) for c in col_names]), file=fout)


info = {
    'time_col': 'date',
    'tables': {},
    'relations': [
      [
        "user:user_id",
        "review:user_id",
        "tip:user_id",
        "user:friends",
      ], [
        "business:business_id",
        "review:business_id",
        "tip:business_id",
      ]
    ]
}
conv_review(info)
conv_business(info)
conv_user(info)
conv_tip(info)
with open(os.path.join(output_dir, 'info.json'), 'w') as f:
  json.dump(info, f, indent=4)
