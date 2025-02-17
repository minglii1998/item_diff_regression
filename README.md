# Item Difficulty Prediction (Regression)

## Packages

I think the main packages used in the code are `torch` and `transformers`, others can be installed one by one. 

## Data

In the data folder, I uploaded a lot of versions. Detailed descriptions can be found in `code_preprocess/preprocess.py`. However, in most cases, you just need to use `data/train_final_type3.json` and `data/test_final_type3.json`.

## Training scripts

Example training scripts can be found in `train_scripts_examples`. 

Typically, you only need to care about `train_bert.py`, `train_bert_nlpaug.py`, and `test_bert.py`. 
