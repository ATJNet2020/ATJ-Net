import gc
import os
import time
import torch
import argparse
import CONSTANT
from atj import atj0_default_param, atj1_default_param, atj2_default_param
from atj import atj0_bayes_opt, atj1_bayes_opt, atj2_bayes_opt
from atj import atj1_random_bayes_opt, atj2_random_bayes_opt
from tree import lgbm_default_param, lgbm_bayes_opt
from database import Database
from utils import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--database_paths', type=str, default='', help='Database paths')
  parser.add_argument('--preprocess', type=str, default='proper', help="basic/simple/proper/full/complex")
  parser.add_argument('--mode', type=str, default='test', help="strategy mode")
  parser.add_argument('--cpu', type=int, default=CONSTANT.JOBS, help="Which gpu to use")
  parser.add_argument('--gpu', type=int, default=0, help="Which gpu to use")
  parser.add_argument('--keval', type=int, default=5, help='Cross evaluation number')
  args, _ = parser.parse_known_args()

  database_paths = args.database_paths.split(',')
  assert len(database_paths) != 0, 'database_paths is required.'
  CONSTANT.JOBS = args.cpu
  CONSTANT.DEVICE = 'cuda:' + str(args.gpu)
  if args.mode == 'test':
    # args.mode = 'atj2_random_bayes_opt'
    args.mode = 'atj1_default_param'
    test_flag = True
  else:
    test_flag = False
  assert args.mode in [
      'atj0_default_param', 'atj1_default_param', 'atj2_default_param',
      'atj0_bayes_opt', 'atj1_bayes_opt', 'atj2_bayes_opt',
      'atj1_random_bayes_opt', 'atj2_random_bayes_opt',
      'lgbm_default_param', 'lgbm_bayes_opt', 'kdd_winner'], 'invalid mode'

  model_func = eval(args.mode)
  total_start_time = time.time()
  results, stats = [], []
  strategy_name = f'{args.preprocess}_{args.mode}' if not test_flag else 'test'
  torch.set_num_threads(CONSTANT.JOBS)

  for database_path in args.database_paths.split(','):
    database_path = os.path.expanduser(database_path)
    database_name = os.path.basename(database_path.rstrip(os.path.altsep))
    output_path = os.path.join(CONSTANT.ROOT_PATH, database_name, strategy_name)
    print('Output to dir:', output_path)
    os.makedirs(output_path, exist_ok=True)

    with Logger(output_path, git_log=True) as logger:
      database = Database(database_path, output_path)
      database.preprocess(args.preprocess)
      database.stat(brief=True)
      database.save_data()
      database.freeze()
      percent_auc, stat = model_func(database, output_path, keval=args.keval)
      print(stat)

    stats.append(stat)
    results.append(percent_auc)
    del database
    gc.collect()

  for stat in stats:
    print(stat)
  print(strategy_name, end='')
  for percent_auc in results:
    print('\t%.4f%% (%.2f%%)' % (percent_auc.mean(), percent_auc.std()), end='')
  t_number, t_metric = time_human_format(time.time() - total_start_time)
  print('\nTotal cost time: %.1f%s' % (t_number, t_metric))
