# Deep-learning Real/Bogus classification for the Tomo-e Gozen transient survey

This repository contains the Python scripts for the experiments in the paper:

Ichiro Takahashi, Ryo Hamasaki, Naonori Ueda, Masaomi Tanaka, Nozomu Tominaga, Shigeyuki Sako, Ryou Ohsawa, Naoki Yoshida, Deep-learning real/bogus classification for the Tomo-e Gozen transient survey, Publications of the Astronomical Society of Japan, 2022;, psac047, https://doi.org/10.1093/pasj/psac047


## files

    ├── README.md
    ├── data
    │   └── raw
    │       └── real_bogus1 <- directory for training data
    └── src
        ├── data
        │   └── make_record.py  <- script for converting training data
        ├── models
        │   ├── real_bogus
        │   │   └── model2.py
        │   └── block.py
        └── real_bogus.py   <- script for training classifiers



## requirements
- python = 3.8
- numpy
- scipy
- pandas
- scikit-learn
- tqdm
- tensorflow >= 2.2
- sacred = 0.8.2

## preparation
### downloading dataset
The dataset is available in https://doi.org/10.5281/zenodo.6691804.
After downloading, move the training data files (images*.npy and params*.csv) to the directory `data/raw/real_bogus1`.

### converting dataset
Only the TFRecord format is supported as training data.  
The following command converts images (images*.npy) and labels (params*.csv) to the TFRecord format.

```bash
cd src/data # move from the project root

# converting training data
python make_record.py
```

The converted training data are generated in `data/processed/real_bogus1`.

### finding label errors
The first stage of training is to find label errors.

```bash
cd src # move from the project root

TRAIN_DATA_DIR=../data/processed/real_bogus1

TEST_DATA_DIR=...
TEST_REAL_NAME=...  # npy file
TEST_BOGUS_NAME=...  # npy file

FILTER_RESULT_DIR=...   # to save the results

# to find label errors
for fold in {0..4}; do
    python real_bogus.py with train_data_dir=${TRAIN_DATA_DIR} \
        test_data_dir=${TEST_DATA_DIR} \
        test_real_name=${TEST_REAL_NAME} test_bogus_name=${TEST_BOGUS_NAME} \
        filter_result_dir=${FILTER_RESULT_DIR} \
        train_cfg.target_fold=${fold}
done
```

The results are generated to `filter_result_dir`

## training
### Comparison of training data treatments
#### Simple-each

```bash
OUTPUT_DIR=...

for id in (list of detector ids); do
    python real_bogus.py with train_data_dir=${TRAIN_DATA_DIR} \
        test_data_dir=${TEST_DATA_DIR} \
        test_real_name=${TEST_REAL_NAME} test_bogus_name=${TEST_BOGUS_NAME} \
        output_dir=${OUTPUT_DIR} \
        filter_result_dir=${FILTER_RESULT_DIR} \
        seed=0 train_cfg.selection_mode=2 \
        train_cfg.lambda_ce=1.0 train_cfg.lambda_auc=0.0 train_cfg.lambda_vat=0.0 \
        train_cfg.optimizer=adadelta batch_size=64 \
        model_cfg.model_type=simple model_cfg.padding=valid \
        model_cfg.detector_id=${id} train_cfg.epochs=1000000
done
```

#### Simple-mix


```bash
OUTPUT_DIR=...

python real_bogus.py with train_data_dir=${TRAIN_DATA_DIR} \
    test_data_dir=${TEST_DATA_DIR} \
    test_real_name=${TEST_REAL_NAME} test_bogus_name=${TEST_BOGUS_NAME} \
    output_dir=${OUTPUT_DIR} \
    filter_result_dir=${FILTER_RESULT_DIR} \
    seed=0 train_cfg.selection_mode=2 \
    train_cfg.lambda_ce=1.0 train_cfg.lambda_auc=0.0 train_cfg.lambda_vat=0.0 \
    train_cfg.optimizer=adadelta batch_size=64 \
    model_cfg.model_type=simple model_cfg.padding=valid \
    train_cfg.small_mix_dataset=True train_cfg.epochs=1000000
```

#### Simple-all

```bash
OUTPUT_DIR=...

python real_bogus.py with train_data_dir=${TRAIN_DATA_DIR} \
    test_data_dir=${TEST_DATA_DIR} \
    test_real_name=${TEST_REAL_NAME} test_bogus_name=${TEST_BOGUS_NAME} \
    output_dir=${OUTPUT_DIR} \
    filter_result_dir=${FILTER_RESULT_DIR} \
    seed=0 train_cfg.selection_mode=2 \
    train_cfg.lambda_ce=1.0 train_cfg.lambda_auc=0.0 train_cfg.lambda_vat=0.0 \
    train_cfg.optimizer=adadelta batch_size=64 \
    model_cfg.model_type=simple model_cfg.padding=valid \
    train_cfg.epochs=1000000
```

### Comparison of label error handling
#### Simple

```bash
OUTPUT_DIR=...

python real_bogus.py with train_data_dir=${TRAIN_DATA_DIR} \
    test_data_dir=${TEST_DATA_DIR} \
    test_real_name=${TEST_REAL_NAME} test_bogus_name=${TEST_BOGUS_NAME} \
    output_dir=${OUTPUT_DIR} \
    filter_result_dir=${FILTER_RESULT_DIR} \
    seed=0 train_cfg.selection_mode=2 \
    train_cfg.lambda_ce=1.0 train_cfg.lambda_auc=0.0 train_cfg.lambda_vat=0.0 \
    model_cfg.model_type=simple model_cfg.padding=valid \
    train_cfg.epochs=1000000
```

#### Complex

```bash
OUTPUT_DIR=...

data_type=0  # removed, unlabeled
# data_type=2  # all
python real_bogus.py with train_data_dir=${TRAIN_DATA_DIR} \
    test_data_dir=${TEST_DATA_DIR} \
    test_real_name=${TEST_REAL_NAME} test_bogus_name=${TEST_BOGUS_NAME} \
    output_dir=${OUTPUT_DIR} \
    filter_result_dir=${FILTER_RESULT_DIR} \
    seed=0 train_cfg.selection_mode=${data_type} \
    train_cfg.lambda_ce=${lambda_ce} train_cfg.lambda_auc=${lambda_ech} train_cfg.lambda_vat=${lambda_lds} \
    model_cfg.model_type=complex model_cfg.norm_type=simple
```

## prediction

```bash
OUTPUT_DIR=...  # specified in training

python real_bogus.py predict --unobserved with \
    test_data_dir=${TEST_DATA_DIR} \
    test_real_name=${TEST_REAL_NAME} test_bogus_name=${TEST_BOGUS_NAME} \
    output_dir=${OUTPUT_DIR}
```

## known issues
- The execution requires enough memory (main memory, not GPU memory) to load all the training data.
- There are some cases where saved models cannot be loaded when an environment different from training or a different version of TensorFlow are used.
