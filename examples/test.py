from transformers import BartTokenizer, BartForSequenceClassification
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = BartTokenizer.from_pretrained('bart-large')
model = BartForSequenceClassification.from_pretrained('bart-large')
model = model.cuda()
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute",
                                          add_special_tokens=True)).unsqueeze(0)  # Batch size 1
input_ids = input_ids.cuda()
labels = torch.tensor([1]).unsqueeze(0).cuda()  # Batch size 1
outputs = model(input_ids, labels=labels)
loss, logits = outputs[:2]