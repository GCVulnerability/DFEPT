# Improving Pre-Training Model Based Vulnerability Detection with Data Flow Embedding

## Requirements
- Python 3.8
- pytorch 2.0.0
- transformers 4.36.2
- tree-sitter 0.20.4
- scikit-learn

## RQ1 Efficiency
### Fine-tune CodeBERT/UnixCoder

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

### Fine-tune GraphCodeBERT
```shell
cd GraphCodeBERT+DFG
python run.py \
    --output_dir=./checkpoint \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --epoch 10 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/train.log
```
### Fine-tune CodeT5
```shell
cd CodeT5+DFG/code
python run_exp.py --model_tag codet5_base --task defect --sub_task none
python run_exp.py --model_tag codet5_base --task defect_reveal --sub_task none
```
## RQ2 Different GNN/Pooling Methods
```shell
cd DifferentGNN
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/unixcoder-base \
    --model_name_or_path=microsoft/unixcoder-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

## Ablation Study
```shell
cd ablation
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/unixcoder-base \
    --model_name_or_path=microsoft/unixcoder-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```


