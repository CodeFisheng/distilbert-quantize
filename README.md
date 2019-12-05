# distilbert-quantize

Download the SQuAD train file [here](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json) and dev file [here](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
To run the training procedure:

```
python custom_run_squad.py \
  --model_name_or_path distilbert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
  --config_name config.json
```
