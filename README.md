# Rainfall Forecasting Using Multi-Graph Convolutional Networks
### Requirements
- Python 3.8
- Tensorflow 2.6.0
- Spektral 1.0.8

### 1. To get the embedded datasets we used
https://drive.google.com/drive/folders/1RlGxlvrGiIxEehR2Fu_-QSBoNxtbkSr8?usp=sharing
https://drive.google.com/drive/folders/1dj3EnPrcKaaB524niNuN5dDNuof1-OZE?usp=sharing

Put data_neg and data_pos under the neg and pos respectively.

### 2. To get the original dataset we used
https://drive.google.com/file/d/10I5eaUixIVQlEz-mlB-zMWG5ViMIT2IX/view?usp=sharing

### 3. Generate the embedding matrix and perform training-testing partition
Run gendata.py under ./MGCN/neg or ./MGCN/pos

### 4. Test the M-GCN under ./MGCN/neg or ./MGCN/pos
Run MGCN.py for training models

Run MGCNtest.py for testing

### Sidenote: Using the GPU
By default, tensorflow is multiple-GPU friendly and it automatically distributes the loads. However, you can also manually lock your computation to one or more of the GPUs. (https://www.tensorflow.org/programmers_guide/using_gpu)
