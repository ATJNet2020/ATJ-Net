import os
import time
import copy
import numpy as np
from model import BaseModel
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

iteration = None
def bayes_opt(database, output_dir, model_class, space,
              max_evals=50, param_func=None, keval=2):
  global iteration
  iteration = 0
  with open(os.path.join(output_dir, 'hyperopt.csv'), 'w') as f:
    f.write(','.join(list(map(str, ['loss', 'params', 'iteration', 'auc',
        'auc.mean', 'auc.std', 'run_time', 'test_aucs']))) + '\n')

  def objective(param):
    global iteration
    iteration += 1
    if param_func is not None:
      param_func(param)
    print(param)

    start_time = time.time()
    model = model_class(
        database, os.path.join(output_dir, str(iteration)),
        keval=keval, **param)
    auc = model.train()
    test_aucs, test_preds = model.predict()
    run_time = time.time() - start_time
    loss = 1 - auc.mean()

    with open(os.path.join(output_dir, 'hyperopt.csv'), 'a') as f:
      f.write(','.join(list(map(str, [loss, param, iteration, auc,
          auc.mean(), auc.std(), run_time, test_aucs]))) + '\n')
    del model
    print(auc, auc.mean(), auc.std())
    return {
        'loss': loss,
        'param': copy.deepcopy(param),
        'iteration': iteration,
        'train_time': run_time,
        'auc': auc,
        'status': STATUS_OK,
        'test_aucs': test_aucs,
        'test_preds': test_preds,
        }

  trials = Trials()
  fmin(
      fn=objective, space=space, algo=tpe.suggest,
      max_evals=max_evals, rstate=np.random.RandomState(50),
      trials=trials, show_progressbar=False)

  ypred = np.stack([v['test_preds'] for v in trials.results], axis=0)
  ypred = ypred.mean(axis=(0,1)).astype(np.float32)
  model = BaseModel(database, output_dir)
  ylabel = model.label[model.test_idx]
  assert ylabel.shape == ypred.shape, \
      'ylabel: %s\typred: %s' % (ylabel.shape, ypred.shape)
  ensemble_auc = roc_auc_score(y_true=ylabel, y_score=ypred)
  print('Ensemble AUC: %.4f%%' % (ensemble_auc*100))

  test_auc = np.stack([v['test_aucs'] for v in trials.results], axis=0)
  print('Best Test AUC: %.4f%%\tlocate at' % (test_auc.max()*100),
    np.unravel_index(test_auc.argmax(), test_auc.shape))

  best_iter = sorted(trials.results, key=lambda x: x['loss'])[0]
  print('AUC on test should be', best_iter['test_aucs'].mean())

  best_param = best_iter['param']
  return best_param
