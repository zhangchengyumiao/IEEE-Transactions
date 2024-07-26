# Unsupervised Action Recognition using Spatiotemporal, Adaptive, and Attention-guided Refining-Network
The full source code is temporarily for interview display only

# Abstract
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


![image](https://github.com/zhangchengyumiao/IEEE-Transactions/assets/125729198/19ef00f0-00f6-4273-87a2-1fbd82fc421e)
![image](https://github.com/zhangchengyumiao/IEEE-Transactions/assets/125729198/1c3c9919-b8d4-4a9d-a3da-5d97f2975362)
![image](https://github.com/zhangchengyumiao/IEEE-Transactions/assets/125729198/a1301d87-8c8b-4ef2-9af2-8ca7e2f9cd08)
![image](https://github.com/zhangchengyumiao/IEEE-Transactions/assets/125729198/a86c0cc1-3981-4de0-9bfa-c610b82be5b7)
