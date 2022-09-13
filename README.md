# Learn to Understand Negation in Video Retrieval

This is the official source code of our paper: Learn to Understand Negation in Video Retrieval.
## Requirements
We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.
```
conda create -n py37 python==3.7 -y
conda activate py37
git clone git@github.com:ruc-aimc-lab/nT2VR.git
cd nT2VR
pip install -r requirements.txt
```

## Prepare Data

### Download official video data

- For MSRVTT, the official data can be found in [link](http://ms-multimedia-challenge.com/2017/dataset).
The raw videos can be found in sharing from [Frozen️ in Time](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip).

  We follow the official MSRVTT3k split and MSRVTT1k split (described in the paper  [JSFUSION](https://arxiv.org/abs/1808.02559))

- For vatex, the official data can be found in this [link](https://eric-xw.github.io/vatex-website/download.html)

  We follow the split of [HGR](https://github.com/cshizhe/hgr_v2t)

- We extract frames from the video at a frame rate of 0.5s before training, using scrip from [video-cnn-feat](https://github.com/xuchaoxi/video-cnn-feat). Each data folder should also contain a file indicates frame id and the image path.(See the example of [id.imagepath.txt](https://pan.baidu.com/s/1E7wUG680kXIHejcsQ0wn0w?pwd=5f86). The prefix of frame id should be consistent with video id.)

### Download text data for Training & Evaluation in nT2V
Download [data](https://pan.baidu.com/s/14awvWfhitDvF3CVNKbcA5Q?pwd=5m34)  for training & evaluation in nT2V.
We use the prefix "msrvtt10k" and "msrvtt1kA" to distinguish  MSR-VTT3k split and MSR-VTT1k split. 
- The training data augumented by negator is named as "\*\*.caption.neagtion.txt". The negated and composed test query sets are named as "\*\*.negated.txt" and "\*\*.composed.txt".

## Evaluation on test queries of nT2V
We provide script for evaluting zero-shot CLIP, CLIP* and CLIP-bnl on nT2V.
+ CLIP: original model, used in a zero-shot setting
+ CLIP*: Fine-tuned CLIP on text-to-video retrieval data using retrieval loss.
+ CLIP-*bnl*: Fine-tuned CLIP using proposed negation leraning.
Here are the  checkpoints and performances of CLIP, CLIP* and CLIP-bnl:

### MSR-VTT3k
| Model Checkpoint| Original |       |       |        |   Negated   |             |              |              | Composed |       |        |        | 
|-----------------|:--------:|:-----:|:-----:|:------:|:-----------:|:-----------:|:------------:|:------------:|:--------:|------:|-------:|-------:|
|                 |     $R1$ |  $R5$ | $R10$ |  $MIR$ | $\Delta R1$ | $\Delta R5$ | $\Delta R10$ | $\Delta MIR$ |     $R1$ |  $R5$ |  $R10$ |  $MIR$ |            
| [CLIP](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)            |    20.8  | 40.3  | 49.7  | 0.305  |        1.5  |        2.5  |         2.9  |       0.020  |     6.9  | 24.2  |  35.6  | 0.160  |            
| [CLIP*](https://pan.baidu.com/s/1FzcuhoQQLlfhpEQeLfXUOg?pwd=ekdd)          |    27.7  | 53.0  | 64.2  | 0.398  |        0.5  |        1.1  |         1.1  |       0.008  |    11.4  | 33.3  |  46.2  | 0.225  |            
| CLIP (boolean)  |       -- |    -- |    -- |     -- |       18.8  |       37.5  |        46.2  |         5.9  |    16.7  | 23.9  | 0.118  | 0.116  |            
| CLIP* (boolean) |       -- |    -- |    -- |     -- |       25.3  |       47.1  |        56.1  |        13.5  |    33.7  | 45.5  | 0.236  | 0.243  |            
| [CLIP-bnl](https://pan.baidu.com/s/13mi2tqrx5q4W_9R-uFPHGQ?pwd=pyeu)      |    28.4  | 53.7  | 64.6  | 0.404  |        5.0  |        6.9  |         6.9  |       0.057  |    15.3  | 40.0  |  53.3  | 0.274  |            

### MSR-VTT1k
| Model Checkpoint| Original |       |       |        |   Negated   |             |              |              | Composed |       |       |        | 
|-----------------|:--------:|:-----:|:-----:|:------:|:-----------:|:-----------:|:------------:|:------------:|:--------:|------:|------:|-------:|
|                 |     $R1$ |  $R5$ | $R10$ |  $MIR$ | $\Delta R1$ | $\Delta R5$ | $\Delta R10$ | $\Delta MIR$ |     $R1$ |  $R5$ | $R10$ |  $MIR$ |            
| CLIP            |    31.6  | 54.2  | 64.2  | 0.422  |         1.4 |         1.4 |          1.5 |        0.017 |    12.9  | 35.0  | 46.2  | 0.237  |            
| [CLIP*](https://pan.baidu.com/s/1ewXs-bIEacFO1vx8E_f_2w?pwd=mdh5)           |    41.1  | 69.8  | 79.9  | 0.543  |        0.0  |        1.7  |         1.0  |       0.006  |    17.3  | 46.8  | 61.2  | 0.310  |            
| CLIP (boolean)  |       -- |    -- |    -- |     -- |       26.4  |       46.2  |        56.8  |        0.354 |     6.3  | 18.4  | 25.9  | 0.129  |            
| CLIP* (boolean) |       -- |    -- |    -- |     -- |       35.9  |       59.5  |        65.2  |       0.463  |    17.6  | 42.0  | 52.0  | 0.291  |            
| [CLIP-bnl](https://pan.baidu.com/s/1Zt7NTo6h58ZHq1XmvGPQow?pwd=pnn2)        |    42.1  | 68.4  | 79.6  | 0.546  |       12.2  |       11.7  |        14.4  |       0.121  |    24.8  | 57.6  | 68.8  | 0.391  |            

### VATEX
| Model Checkpoint| Original |       |       |        |   Negated   |             |              |              | Composed |       |       |        | 
|-----------------|:--------:|:-----:|:-----:|:------:|:-----------:|:-----------:|:------------:|:------------:|:--------:|------:|------:|-------:|
|                 |     $R1$ |  $R5$ | $R10$ |  $MIR$ | $\Delta R1$ | $\Delta R5$ | $\Delta R10$ | $\Delta MIR$ |     $R1$ |  $R5$ | $R10$ |  $MIR$ |            
| CLIP            |    41.4  | 72.9  | 82.7  | 0.555  |         1.9 |         2.1 |          2.2 |        0.018 |    10.5  | 28.3  | 41.3  | 0.201  |            
| [CLIP*](https://pan.baidu.com/s/1O57EKap5QJ9TC0isd_ljBQ?pwd=7a37)           |    56.8  | 88.4  | 94.4  | 0.703  |        0.2  |        0.4  |         0.7  |       0.004  |    14.2  | 39.2  | 53.3  | 0.266  |            
| CLIP (boolean)  |       -- |    -- |    -- |     -- |       32.5  |       57.2  |        64.5  |       0.431  |     5.0  | 18.0  | 25.6  | 0.116  |            
| CLIP* (boolean) |       -- |    -- |    -- |     -- |       25.3  |       47.1  |        56.1  |       0.353  |    14.1  | 34.4  | 45.1  | 0.243  |            
| [CLIP-bnl](https://pan.baidu.com/s/18Ft3Guc077_Slht9pUa2sQ?pwd=q9rj)        |    57.6  | 88.3  | 94.0  | 0.708  |       14.0  |       11.7  |         8.6  |       0.125  |    16.6  | 39.9  | 53.9  | 0.284  |            

- To evaluate zero-shot CLIP, run the script [clip.sh](shell/test/clip.sh)
```sh
# use 'rootpath' to specify the path to the data folder
cd shell/test
bash clip.sh
```

- To evaluate CLIP*, run the script [clipft.sh](shell/test/clip.sh)
```sh
# use 'rootpath' to specify the path to the data folder
# use 'model_path' to specify the path of model
cd shell/test
bash clipft.sh
```

- To evaluate zero-shot CLIP+boolean, run the script [clip_bool.sh](shell/test/clip_bool.sh)
```sh
cd shell/test
bash clip_bool.sh
```
- To evaluate CLIP*+boolean, run the script [clipft_bool.sh](shell/test/clip_bool.sh)
```sh
cd shell/test
bash clipft_bool.sh
```
- To evaluate CLIP-bnl, run the script [clip_bnl.sh](shell/test/clip.sh)
```sh
cd shell/test
bash clip_bnl.sh
```

## Train CLIP-bnl from scratch
- train CLIP-bnl on MSR-VTT3k split, run
```sh
# use 'rootpath' to specify the path to the data folder
cd shell/train
bash msrvtt7k_clip_bnl.sh
```
- train CLIP-bnl on MSR-VTT1k split, run
```sh
cd shell/train
bash msrvtt9k_clip_bnl.sh
```
- train CLIP-bnl on VATEX, run
```sh
cd shell/train
bash vatex_clip_bnl.sh
```
- Additionally, training script of CLIP* is [clipft.sh](shell/train/clipft.sh) 


## Produce new negated & composed data
1.  install additional packages:
```sh
cd negationdata
pip install -r requirements.txt
```
2. download [checkpoint](https://pan.baidu.com/s/1KwKENCE9NSKvQ2VNi0UOcA?pwd=6mj6) of negation scope detection model,which is built on [NegBERT](https://github.com/adityak6798/Transformers-For-Negation-and-Speculation)
3. run the script [prepare_data.sh](negationdata/prepare_data.sh)
```sh
# use 'rootpath' to specify the path to the data folder
#use 'cache_dir'to specify the path to path of models used in negation scope detection model 
cd negationdata
bash prepare_data.sh
```

## Citation
```
@inproceedings{mm22-nt2vr,
title = {Learn to Understand Negation in Video Retrieval},
author = {Ziyue Wang and Aozhu Chen and Fan Hu and Xirong Li},
year = {2022},
booktitle = {ACMMM},
}
```

## Contact
If you enounter issues when running the code, please feel free to reach us.
- Ziyue Wang (ziyuewang@ruc.edu.cn)
- Aozhu Chen (caz@ruc.edu.cn)
- Fan Hu (hufan_hf@ruc.edu.cn)
