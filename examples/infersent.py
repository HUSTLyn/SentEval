# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging

# get models.py from InferSent repo
from models import InferSent

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = 'data/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = 'data/infersent1.pkl'
V = 1 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(PATH_TO_W2V)

    params_senteval['infersent'] = model.cuda()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']

    results = se.eval(transfer_tasks)

    sts_task_list = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    other_task_list = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKEntailment']

    # STS pearson
    print("######### STS pearson ###########")
    summ = 0
    s_title = ''
    s_result = ''
    for task in sts_task_list:
        try:
            task_result = results[task]['all']['pearson']['mean'] * 100
        except:
            task_result = results[task]['pearson'] * 100
        s_title += '%6s' % (task[:5])
        s_result += ' %.2f' % (task_result)
        summ += task_result
    print(s_title)
    print(s_result)
    print('avg: %.2f' % (summ / len(sts_task_list)))

    # STS spearman
    print("######### STS spearman ###########")
    summ = 0
    s_title = ''
    s_result = ''
    for task in sts_task_list:
        try:
            task_result = results[task]['all']['spearman']['mean'] * 100
        except:
            task_result = results[task]['spearman'] * 100
        s_title += '%6s' % (task[:5])
        s_result += ' %.2f' % (task_result)
        summ += task_result
    print(s_title)
    print(s_result)
    print('avg: %.2f' % (summ / len(sts_task_list)))

    # Other
    print("######### Other ###########")
    summ = 0
    s_title = ''
    s_result = ''
    for task in other_task_list:
        task_result = results[task]['acc']
        s_title += '%6s' % (task[:5])
        s_result += ' %.2f' % (task_result)
        summ += task_result
    print(s_title)
    print(s_result)
    print('avg: %.2f' % (summ / len(other_task_list)))

