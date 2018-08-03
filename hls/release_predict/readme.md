## Data preprocessing
python preprocess.py  [-h] [--data_dir DATA_DIR]

```
optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Directory to the input dataset.
```

## Model training
python train.py [-h] [--data_dir DATA_DIR] [--save_model_dir SAVE_MODEL_DIR] [--feature_select]

```
optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory to the training dataset.
  --save_model_dir SAVE_MODEL_DIR
                        Directory to save the trained model. Input folder or
                        file name.
  --feature_select      Use feature selection.
```

## Model testing
python test.py [-h] [--data_dir DATA_DIR] [--model_dir MODEL_DIR] [--save_result_dir SAVE_RESULT_DIR]

```
optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory to the testing dataset.
  --model_dir MODEL_DIR
                        Directory to the pre-trained models.
  --save_result_dir SAVE_RESULT_DIR
                        Directory to save the result. Input folder or file
                        name.
```