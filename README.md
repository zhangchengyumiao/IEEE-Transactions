# Unsupervised Action Recognition using Spatiotemporal, Adaptive, and Attention-guided Refining-Network
The full source code is temporarily for interview display only

## Abstract
Previous works on unsupervised skeleton-based ac
tion recognition primarily focused on strategies for utilizing
 features to drive model optimization through methods like
 contrastive learning and reconstruction. However, designing
 application-level strategies poses challenges. This paper shifts
 the focus to the generation-level modelings and introduces the
 Spatiotemporal Adaptively Attentions-guided Refining Network
 (AgRNet). AgRNet approaches the reduction of costs and en
hancement of efficiency by constructing the Adaptive Activity
Guided Attention (AAGA) and Adaptive Dominant-Guided At
tenuation (ADGA) modules. The AAGA leverages the sparsity of
 the correlation matrix in the attention mechanism to adaptively
 f
 ilter and retain the active components of the sequence during the
 modeling process. The ADGA embeds the local dominant features
 of the sequence, obtained through convolutional distillation, into
 the globally dominant features under the attention mechanism,
 guided by the defined attenuation factor. Additionally, the
 Progressive Feature Modeling (PFM) module is introduced to
 complement the progressive features in motion sequences that
 were overlooked by AAGA and ADGA. AgRNet shows efficiency
 on three public datasets, NTU-RGBD 60, NTU-RGBD 120, and
 UWA3D.
 Impact Statement—As a time series with clear physical signif
icance, motion sequences possess unique spatiotemporal entan
glement characteristics. Efficiently modeling these features is not
 only fundamental but also challenging in motion-related tasks.
 However, recent unsupervised action recognition paradigms have
 predominantly focused on the application level of features,
 neglecting the underlying modeling approach. This paper intro
duces AgRNet, situated in the domain of unsupervised skeleton
 recognition tasks. The primary concept optimizes traditional
 attention mechanisms for more efficient modeling of skeleton
 sequences, employing three modules. AgRNet demonstrates ver
satility, showcasing state-of-the-art performance in various time
series tasks, including skeleton data. This capability is distinctly
 validated in the ablation experiments section of this paper.
 Index Terms—Spatiotemporal modeling, Unsupervised action
 recognition, Attention mechanism
 
## Architecture Overview.
 ![image](https://github.com/user-attachments/assets/2228e747-9829-452a-9922-7ad53b24e01a)
 
## Requirements
1. Tensorflow 1.14.0\
2. Python 3\ 
3. torch == 1.8.0 \
4. scikit-learn 0.21.2\
5. matplotlib 3.1.0\
6. numpy 1.16.4\

## Date
### NTU RGB+D 60:
The dataset contains 60 classes, with a
total of 56880 samples. They were performed by 40 subjects
aged from 10 to 35. The data set was captured by Microsoft’s
Kinectv2 sensor using three different camera angles in the
form of depth information, 3D skeleton information, RGB
frames, and infrared sequences. The NTU dataset uses two
benchmarks when dividing the training and test sets. Crosssubject divides the training set and test set by person ID. The
training set contains 40,320 samples, and the test set contains
16,560 samples. Cross-View divides the training set and the
test set by the camera. The samples collected by camera 1
are used as the test set, and cameras 2 and 3 are used as
the training set. The sample numbers are 18,960 and 37,920,
respectively. We test our method on both cross-view and crosssubject protocols.
### NTU RGB+D 120: 
This dataset extends NTU RGB+D 60,
a total of 113,945 samples over 120 classes performed by
106 volunteers and captured with 32 different camera setups.
There are a large number of samples with missing bones in
the data, and we deleted them as required during training and
testing. The original paper on this dataset recommends two
benchmarks: (1) the cross-subject (X-Sub) benchmark and (2)
the cross-setup (X-Setup) benchmark, on which we test our
method.
### Multiview Activity II (UWA3D):
The dataset comprises
30 human actions, each performed 4 times by 10 subjects.
It records 15 joints, and each action is observed from four
different views: frontal, left and right sides, and top. The
dataset presents challenges due to the multitude of views,
leading to self-occlusions when considering only parts of them.
Additionally, there is a high similarity among actions; for
instance, the actions ”drinking” and ”phone answering” have
many nearly identical nodes. Even in the dynamic key points,
subtle differences exist.



![image](https://github.com/zhangchengyumiao/IEEE-Transactions/assets/125729198/19ef00f0-00f6-4273-87a2-1fbd82fc421e)
![image](https://github.com/zhangchengyumiao/IEEE-Transactions/assets/125729198/1c3c9919-b8d4-4a9d-a3da-5d97f2975362)
![image](https://github.com/zhangchengyumiao/IEEE-Transactions/assets/125729198/a1301d87-8c8b-4ef2-9af2-8ca7e2f9cd08)
![image](https://github.com/zhangchengyumiao/IEEE-Transactions/assets/125729198/a86c0cc1-3981-4de0-9bfa-c610b82be5b7)
