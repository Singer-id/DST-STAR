## Usage

### Environment
+ Pytorch 1.9.0
+ Transformers 4.24.0

### Data Preprocessing
#### use 2.1 dataset
```
mkdir data
python create_data.py 
python preprocess_data.py
python convert_by_turn.py
```
#### use 2.4 dataset
```
python create_data.py --main_dir data/mwz24 --mwz_ver 2.4 --target_path data/mwz2.4
python preprocess_data.py --data_dir data/mwz2.4
python convert_by_turn.py --data_dir data/mwz2.4
```

### Train and Evaluate
#### use 2.1 datasets
```
#train and test with power-law-attention
python train_STAR.py --save_dir out-bert/model_name 
#test with candidate strategy
python evaluation.py --save_dir out-bert/model_name 
```
#### use 2.4 dataset
```
#train and test with power-law-attention
python train_STAR.py --save_dir out-bert/model_name --data_dir data/mwz2.4
#test with candidate strategy
python evaluation.py --save_dir out-bert/model_name --data_dir data/mwz2.4
```

