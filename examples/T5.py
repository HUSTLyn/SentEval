from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import argparse


# set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# import senteval
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'

sys.path.insert(0, PATH_SENTEVAL)
import senteval
import transformers
from transformers import T5Model,T5Config, T5Tokenizer

def prepare(params, samples):
    pass

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    batch = params['tokenizer'].batch_encode_plus(
            sentences,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_tensors='pt',
            return_attention_masks=True)
    input_ids = batch['input_ids'].cuda()
    att_mask = batch['attention_mask'].cuda()
    with torch.no_grad():
        seq = params['model'](input_ids=input_ids,attention_mask=att_mask, decoder_input_ids=input_ids) # (B, L, H)
        seq=seq[0]
        emb = []
        length = (att_mask == 1).sum(-1)
        for i in range(seq.size(0)):
            emb.append(seq[i][1:length[i]-1].mean(0))
        emb = torch.stack(emb)
        # emb = seq[:,0]

    return emb.cpu()

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='t5-large', help='model name or path')
    args = parser.parse_args()

    config = T5Config.from_pretrained(args.model)
    model = T5Model.from_pretrained(args.model,config=config)
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    params_senteval['model'] = model.cuda().eval()
    params_senteval['tokenizer'] = tokenizer

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
    #                   'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion', 'ImageCaptionRetrieval', 'SNLI'
    #                   ]
    transfer_tasks = [ 'SNLI', 'ImageCaptionRetrieval'
                      ]

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

