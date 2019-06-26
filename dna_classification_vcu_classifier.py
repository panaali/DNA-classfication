# -*- coding: utf-8 -*-
"""dna-classification-vcu.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13BLlKCJc5EiKMHP7KEH5hOXyLEIUdX02

# DNA Multi Class Classification
"""


"""## Prepare Google Drive"""

# Run this cell to mount your Google Drive.

local_path = './'

"""## Prepare fastai"""
from fastai import *
from fastai.text import *

"""## Prepare Dataset"""

local_project_path = local_path + 'dna-10class/'

if not os.path.exists(local_project_path):
  os.makedirs(local_project_path)
  
print('local_project_path:', local_project_path)
"""## Create Language Model"""

class dna_tokenizer(BaseTokenizer):
  def tokenizer(slef, t):
    return list(t)

tokenizer = Tokenizer(tok_func=dna_tokenizer, pre_rules=[], post_rules=[], special_cases=[])
processor = [TokenizeProcessor(tokenizer=tokenizer, include_bos= False, include_eos=False), NumericalizeProcessor(max_vocab=30000)]

# batch size
bs = 64

data_lm = TextLMDataBunch.from_csv(local_project_path, 'combined.csv',
                                   text_cols ='Text', valid_pct= 0.1, tokenizer=tokenizer,
                                   include_bos= False, include_eos=False)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, pretrained=False).to_fp16()
learn.load('lm-fine-tuned-10-4')
learn.save_encoder('lm-fine-tuned-10-4-encoder')

# data_lm.train_ds[0][0].text

# data_lm.train_ds[0][0].data

"""## Create Language Model Learner"""

"""## Create Classifier"""

# data_cls = TextClasDataBunch.from_csv(local_project_path, 'combined_train.csv',
#                                    text_cols ='Text', label_cols ='class', valid_pct= 0.1, tokenizer=tokenizer,
#                                    include_bos= False, include_eos=False, vocab = data_lm.vocab)
data_cls = (TextList.from_csv(local_project_path, 'combined.csv', cols='Text', vocab=data_lm.vocab, processor= processor)
                   .split_from_df(col='is_test')
                   .label_from_df(cols='class')
                   .databunch(bs=bs))

print('data_cls validation set size', len(data_cls.valid_ds))
# data_cls.show_batch()

learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.3, pretrained=False).to_fp16()
learn.load_encoder('lm-fine-tuned-10-4-encoder')

# learn.lr_find()
# learn.recorder.plot()

learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.save('first')


learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

learn.save('second')


learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

learn.save('third')


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('forth')

learn.fit_one_cycle(20, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('fifth')

dna_string = 'atggcagtggcaggtaaaaatgactttgcagttctcaacaccgggcggaagatgcctctccttgggctgggaacatggaagagtgaacctggcaaggttaaacaggcagtaatctgggccttgcaggctggctaccgccacttcgactgtgctgccatctatggcaacgagttggagatcggagaagctctgcaggagacacttggccctgacaaagccttgaggcgagaggatgtgtttatcacctccaagctgtggaacacacagcatcacccggaggatgtggagcccgctctgctgaagacactgaaggagctgagtctggaatacctggatctatacctcatccactggccctatgccttccaacaaggtgacgctcctttccccaaatcggaggacggcaccctgctgtacgacgacatcgactacaagctgacttgggctgccatggagaagctggtgggaaagggcctggtcagggctatcggcctgtccaacttcaacagcaaacagatagacaacgttctctccgtagccaacatcaaaccgactgtgcttcaggtggaaagccatccgtatctggctcaggtggagttgctgggacactgccgggacagaggcctggtgattacagcgtacagcccactggggtcaccggatcgggtatggaagcatcctgatgagcccgtcctcctggatgaagcagcaatcgacaccctggccaagaagtacaacaagtccccagcacaaatcatccttagatggcagacacagcgaggagtagtgacgatccctaaaagtgtgacagagtctcggatcaaagagaatattcaggtatttgactttacccttgaagcggaagagatgaagtgtataacagcattgaacagaggctggcgctacattgtaccaaccatcacagttgatgggaagcccgtccccagggatgcaggacatccacactaccccttcagtgacccctactga'
learn.predict(dna_string)

