# BERT4Rec
This is a version of [https://github.com/FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec) ported to Python3 and Tensorflow2. Tested with python3.9 and tesorflow 2.6.0, but likely to work with other python3 and tensorflow 2 versions

If you use this version of the code for your research, please consider citing the [reproducibility paper](https://arxiv.org/pdf/2207.07483.pdf) (the port was done as a part of the reproducibility work): 

```
@inproceedings{petrov2022replicability,
  title={A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation},
  author={Petrov, Aleksandr and Macdonald, Craig},
  booktitle={Sixteen ACM Conference on Recommender Systems},
  year={2022}
}
```
I also recommend to read this paper, and in particular the section regarding the training time required for BERT4Rec convergence. 

Also, consider our more efficient implementation based on Hugging Face trarsformers (https://github.com/asash/bert4rec_repro)


## Usage

**Requirements**

* python 3.9
* Tensorflow 2.6.0 (GPU version)
* CUDA compatible with TF 2.6.0 

**Run**

For simplicity, here we take ml-1m as an example:

``` bash
./run_ml-1m.sh
```
include two part command:
generated masked training data
``` bash
python -u gen_data_fin.py \
    --dataset_name=${dataset_name} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --mask_prob=${mask_prob} \
    --dupe_factor=${dupe_factor} \
    --masked_lm_prob=${masked_lm_prob} \
    --prop_sliding_window=${prop_sliding_window} \
    --signature=${signature} \
    --pool_size=${pool_size} \
```

train the model
``` bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/${dataset_name}${signature}.vocab \
    --user_history_filename=./data/${dataset_name}${signature}.his \
    --checkpointDir=${CKPT_DIR}/${dataset_name} \
    --signature=${signature}-${dim} \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --learning_rate=1e-4
```

### hyper-parameter settings
json in `bert_train` like `bert_config_ml-1m_64.json`

```json
{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 200,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3420
}
```


## References

```TeX
@inproceedings{petrov2022replicability,
  title={A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation},
  author={Petrov, Aleksandr and Macdonald, Craig},
  booktitle={Sixteen ACM Conference on Recommender Systems},
  year={2022}
}

@inproceedings{Sun:2019:BSR:3357384.3357895,
 author = {Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
 title = {BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
 booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '19},
 year = {2019},
 isbn = {978-1-4503-6976-3},
 location = {Beijing, China},
 pages = {1441--1450},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3357384.3357895},
 doi = {10.1145/3357384.3357895},
 acmid = {3357895},
 publisher = {ACM},
 address = {New York, NY, USA}
} 
```
