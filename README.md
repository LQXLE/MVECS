## scMVAF

### Requirements

python \--- 3.9.19
numpy \--- 1.24.4
pandas \--- 1.5.3
scanpy \--- 1.10.0
scikit-learn ---1.4.1.post1
torch \--- 2.0.0+cu118
torchaudio \--- 2.0.1+cu118
torchvision \--- 0.15.1+cu118

### Dataset (`data.mat`)

```
- `data.mat` is a MATLAB-formatted dataset used as input for the model.
- data['X']: Count matrix, where each row represents a cell and each column represents a feature (e.g., gene expression counts).
- data['Y']: Ground truth labels for each cell.
```

### Examples

The datasets folder includes the preprocessed Quake_Smart-seq2_Limb_Muscle dataset, Deng dataset, and PBMC-Zheng4k dataset. To use a different dataset, you need to modify the following code:

```python
parser.add_argument('--data_name', default='goolam')
```

### Run

```
python scMVAF.py
```

