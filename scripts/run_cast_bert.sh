python train.py --data_dir  $data_dir \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--save_path $save_path \
--cache_dir $cache_dir \
--train_file train_annotated.json \
--distant_file train_annotated.json \
--dev_file  dev_revised.json  \
--test_file test_revised.json \
--train_batch_size 4 \
--test_batch_size 4 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--drop_prob 0.0  \
--inference_th 0.9 \
--gamma_p 1.0 \
--gamma_r 1.0 \
--num_train_epochs 10.0  \
--n_splits 5 \
--n_rounds 7 \
--seed 9 \
--num_class 97
