# High-Quality Temporal Link Prediction for Weighted Dynamic Graphs via Inductive Embedding Aggregation

This repository provides a reference implementation of *IDEA* introduced in the paper "High-Quality Temporal Link Prediction for Weighted Dynamic Graphs via Inductive Embedding Aggregation".

### Abstract
Temporal link prediction (TLP) is an inference task on dynamic graphs that predicts future topology using historical graph snapshots. Existing TLP methods are usually designed for unweighted graphs with fixed node sets. Some of them cannot be generalized to the prediction of weighted graphs with non-fixed node sets. Although several methods can still be used to predict weighted graphs, they can only derive low-quality prediction snapshots sensitive to large edge weights but fail to distinguish small and zero weights in adjacency matrices. In this study, we consider the challenging high-quality TLP on weighted dynamic graphs and propose a novel inductive dynamic embedding aggregation (IDEA) method, inspired by the high-resolution video prediction. IDEA combines conventional error minimization objectives with a scale difference minimization objective, which can generate high-quality weighted prediction snapshots, distinguishing differences among large, small, and zero weights in adjacency matrices. Since IDEA adopts an inductive dynamic embedding scheme with an attentive node aligning unit and adaptive embedding aggregation module, it can also tackle the TLP on weighted graphs even with non-fixed node sets. Experiments on datasets of various scenarios validate that IDEA can derive high-quality prediction results for weighted dynamic graphs and tackle the variation of node sets.

### Citing
If you find this project useful for your research, please cite the following paper.
```
@article{qin2023high,
  title={High-Quality Temporal Link Prediction for Weighted Dynamic Graphs Via Inductive Embedding Aggregation},
  author={Qin, Meng and Zhang, Chaorui and Bai, Bo and Zhang, Gong and Yeung, Dit-Yan},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}

```

If you have questions, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
* Numpy
* Scipy
* PyTorch

### Usage

Please download the [datasets](https://hkustconnect-my.sharepoint.com/:u:/g/personal/mqinae_connect_ust_hk/EWhWevRDJ9lNttKte2oowxcBKPWa9aewkGv7i7-FZJvyaQ?e=6teK6g) (~10.7GB), including the (weighted) edge sequence, modularity matrix input, node feature input (if available), node aligning matrix sequence (if for L2 & L3), node set sequence (if for L2 & L3) for each snapshot. Unzip it and put all the data files under the directory ./data.

The IDEA model and its loss functions are defined and implemented by the script in ./modules. In particular, *layers.py*, *IDEA.py*, and *loss.py* are refactored scripts with more readable comments and clear definitions consistent with the paper's final version. *HQ_TLP_v1.py* includes some old implementations for the saved checkpoints.

The checkpoints (w.r.t. all the datasets) are put in ./chpt. Please run the corresponding script *IDEA_[dataset name]_chpt.py* to load the checkpoints and check the results. For *Mesh*, *HMob*, *DC*, *SEvo*, and *IoT*, please **uncomment** line 152 and line 194 (but **comment** line 151 and line 193) in *HQ_TLP_v1.py*. Moreover, for *T-Drive* and *WIDE*, please **uncomment** line 151 and line 193 (but **comment** line 152 and line 194) in *HQ_TLP_v1.py*.

To train the IDEA model from scratch, please run the corresponding script *IDEA_[dataset name]_demo.py*. The result w.r.t. each epoch will be saved under the directory ./res. If the flag variable *save_flag* (in *IDEA_[dataset name]_demo.py*) is set to be **True**, the checkpoint w.r.t. each epoch will be saved under the directory ./pt.
