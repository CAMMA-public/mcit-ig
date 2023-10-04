# MCIT-IG: Surgical Action Triplet Detection by Mixed Supervised Learning of Instrument-Tissue Interactions (MICCAI 2023)

<i>S. Sharma, C. I. Nwoye, D. Mutter, N. Padoy</i>

This is the official implementation of MCIT-IG in PyTorch.

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-f9f107.svg)](https://arxiv.org/abs/2307.09548)

## News
* [05/10/2023] Release of code in PyTorch for training and evaluation.

## Abstract
Surgical action triplets describe instrument-tissue interactions as (instrument, verb, target) combinations, thereby supporting a detailed analysis of surgical scene activities and workflow. This work focuses on surgical action triplet detection, which is challenging but more precise than the traditional triplet recognition task as it consists of joint (1) localization of surgical instruments and (2) recognition of the surgical action triplet associated with every localized instrument. Triplet detection is highly complex due to the lack of spatial triplet annotation. We analyze how the amount of instrument spatial annotations affects triplet detection and observe that accurate instrument localization does not guarantee better triplet detection due to the risk of erroneous associations with the verbs and targets. To solve the two tasks, we propose MCIT-IG, a two-stage network, that stands for Multi-Class Instrument-aware Transformer-Interaction Graph. The MCIT stage of our network models per class embedding of the targets as additional features to reduce the risk of misassociating triplets. Furthermore, the IG stage constructs a bipartite dynamic graph to model the interaction between the instruments and targets, cast as the verbs. We utilize a mixed-supervised learning strategy that combines weak target presence labels for MCIT and pseudo triplet labels for IG to train our network. We observed that complementing minimal instrument spatial annotations with target embeddings results in better triplet detection. We evaluate our model on the CholecT50 dataset and show improved performance on both instrument localization and triplet detection, topping the leaderboard of the CholecTriplet challenge in MICCAI 2022. 

## Model Overview
![MCIT-IG](media/main_model.jpg)


<br>

## Pre-requisities
* Download the CholecT50 dataset from https://github.com/CAMMA-public/cholect50.
* Install [ivtmetrics](https://github.com/CAMMA-public/ivtmetrics) for evaluation.
* For more details on the splits, please refer the paper [Data Splits and Metrics](https://arxiv.org/abs/2204.05235).

<br>

## Libraries required
* dgl     : 1.0.0+cuda102
* pytorch : 1.12.0
* numpy   : 1.21.0

<br>

## Training Details
MCIT-IG has been trained on `Nvidia V100 GPU` with `CUDA version 10.2`. Run the script below to launch training. Currently, the model is adapted to train only on one GPU.
```bash
bash train.sh
```

<br>

## Evaluation 
TBD

<br>

## Acknowledgements
This work was supported by French state funds managed by the ANR within the National AI
Chair program under Grant ANR-20-CHIA-0029-01 (Chair AI4ORSafety) and within the Investments for the future
program under Grant ANR-10-IAHU-02 (IHU Strasbourg). It was also supported by BPI France under reference
DOS0180017/00 (project 5G-OR). It was granted access to the HPC resources of Unistra Mesocentre and GENCI-IDRIS
(Grant AD011013710).

<br>

## License
This code, models, and datasets are available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.


## References
If you find MCIT-IG useful in your research, please use the following BibTeX entry for citation.

```bibtex
@article{sharma2023surgical,
      title={Surgical Action Triplet Detection by Mixed Supervised Learning of Instrument-Tissue Interactions}, 
      author={Saurav Sharma and Chinedu Innocent Nwoye and Didier Mutter and Nicolas Padoy},
      year={2023},
      eprint={2307.09548},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


