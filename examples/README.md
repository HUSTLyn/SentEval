要测试模型xxx时，

```
python xxx.py
```

在代码文件中，在transfer_tasks中添加对应的子任务的名称即可，如：

```
transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                  'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 													'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion', 										'ImageCaptionRetrieval', 'SNLI'
                      ]
```

