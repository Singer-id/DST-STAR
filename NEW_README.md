## Usage
### Data Preprocessing
#### 2.1 dataset
```
python create_data.py 
python preprocess_data.py
python convert_by_turn.py
```
#### 2.4 dataset
```
python create_data.py --main_dir data/mwz24 --mwz_ver 2.4 --target_path data/mwz2.4
python preprocess_data.py --data_dir data/mwz2.4
python convert_by_turn.py
```

### Training

```
❱❱❱ python3 train_STAR.py
```

### Evaluation

```console
❱❱❱ python3 evaluation.py
```

If you don't want to re-train the model from scratch, you can download the saved model_dict from [here](https://drive.google.com/file/d/1Bz86HK4ebLqWlg4bd6voGv5TlT0x2qT6/view?usp=sharing). 

## Citation

```bibtex
@inproceedings{ye2021star,
  title={Slot Self-Attentive Dialogue State Tracking},
  author={Ye Fanghua, Manotumruksa Jarana, Zhang Qiang, Li Shenghui, Yilmaz Emine},
  booktitle={The Web Conference (WWW)},
  year={2021}
  }
```

## Contact

If there are any questions, feel free to contact me at smartyfh@outlook.com.
