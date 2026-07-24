# Q-MCMF++: Generalizable Quality-Guided Matcher for Incremental Detection Transformer

> **Q-MCMF (AAAI 2026)** — **Q-MCMF++ (Extended Version)**  
> Qirui Wu, Shizhou Zhang, De Cheng, Yinghui Xing, Lingyan Ran, Dahu Shi, Peng Wang, Yanning Zhang

This repository contains the official implementation of both **Q-MCMF** and its journal extension **Q-MCMF++**.

## Abstract

Incremental Object Detection is critical for deploying detection systems in dynamic environments, as models must continuously learn novel categories without forgetting prior knowledge. While Detection Transformer (DETR) has emerged as a powerful detection paradigm, it suffers from severe catastrophic forgetting in such incremental settings. In this work, we propose Q-MCMF++, a unified framework that equips DETR with robust continual learning capabilities. Specifically, we identify that the exhaustive Hungarian matching in DETR forces geometrically implausible prediction-target assignments, causing background regions to be incorrectly supervised as foreground, a phenomenon we term *background foregrounding*, which causes catastrophic forgetting. To resolve this, we design a Quality-guided Min-Cost Max-Flow (Q-MCMF) matcher that replaces Hungarian matching with a flow optimization formulation, using quality-guided edge pruning to eliminate implausible matches while preserving one-to-one correspondence and maximizing valid assignments. Furthermore, to handle the stringent non-co-occurrence scenario where old classes are entirely absent from new data, we introduce a prototype replay mechanism that decouples feature aggregation from learnable projections in deformable attention to construct compact class-wise prototypes, and repurposes Q-MCMF in a one-to-many configuration for selective query-prototype matching. Experimental results on the COCO dataset demonstrate that our approach significantly alleviates catastrophic forgetting and achieves state-of-the-art performance across both co-occurrence and non-co-occurrence protocols.

## Overview

We propose a Quality-guided Min-Cost Max-Flow (Q-MCMF) matcher for incremental object detection in DETR-based detectors, and extend it to a unified framework with prototype replay.

<p align="center">
  <img src="resource/main_framework.png" alt="Q-MCMF++ Framework" width="100%"/>
  <br>
  <em>Overall framework of Q-MCMF++. The left branch performs object detection with Q-MCMF matcher replacing Hungarian matching for label assignment. The right branch replays class-wise prototypes via one-to-many Q-MCMF matching.</em>
</p>

- **Background Foregrounding.** We identify that the exhaustive Hungarian matching in DETR forces geometrically implausible prediction-target assignments, causing erroneous supervision that leads to catastrophic forgetting.
- **Q-MCMF Matcher.** We reformulate label assignment as a min-cost max-flow problem with quality-guided edge pruning, eliminating implausible matches while preserving valid one-to-one correspondence.
- **Prototype Replay (Q-MCMF++).** To handle non-co-occurrence scenarios, we introduce a prototype replay mechanism with decoupled feature aggregation. Q-MCMF is repurposed in a one-to-many configuration for selective query-prototype matching.

## Code

Code will be released within two weeks (by **August 7, 2026**).

## Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{wu2026qmcmf,
  title={Q-MCMF: Quality-Guided Min-Cost Max-Flow Matcher for Incremental Detection Transformer},
  author={Wu, Qirui and Zhang, Shizhou and Cheng, De and Xing, Yinghui and Ran, Lingyan and Shi, Dahu and Wang, Peng and Zhang, Yanning},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

