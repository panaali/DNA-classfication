"""## Create Classifier"""

data_cls = TextClasDataBunch.from_csv(local_project_path, 'combined_train.csv',
                                   text_cols ='Text', label_cols ='class', valid_pct= 0.1, tokenizer=tokenizer,
                                   include_bos= False, include_eos=False, vocab = data_lm.vocab)

# data_cls.show_batch()

learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.3, pretrained=False).to_fp16()
learn.load_encoder('fine_tuned_enc')

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

