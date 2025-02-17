
python train_bert.py \
    --train_data_path data/train_final_type3.json \
    --test_data_path data/train_final_type3.json \
    --split_ratio 0.15 \
    --learning_target Difficulty \
    --model_name bert-base-uncased \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --batch_size 16 \
    --num_epochs 10 \
    --evaluation_strategy steps \
    --output_dir ./train_final_type3/BERT_Difficulty_type4_5e6_warm1_decay01_bs16_epo10_split15 \
    --log_dir ./logs 

python test_bert.py \
    --test_data_path data/test_final_type3.json \
    --learning_target Difficulty \
    --model_dir ./train_final_type3/BERT_Difficulty_type4_5e6_warm1_decay01_bs16_epo10_split15 \
    --model_name bert-base-uncased \
    --output_file ./train_final_type3/BERT_Difficulty_type4_5e6_warm1_decay01_bs16_epo10_split15/test_results.json
