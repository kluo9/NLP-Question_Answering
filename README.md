# NLP-Question_Answering
Chinese question answering language model

- Bert_model.py: Fine tune BERT-base pretrained model. Number
of epoch is 2.
Accuracy in validation set is 0.5.
- Bert_model_enhanced.py: Fine tune BERT-base pretrained model.
Added learning rate scheduler and adjusted doc-stride so the 
paragraph in validation/testing are overlapped. Number of epoch
is set to 5.
Accuracy in validation set is 0.56.

