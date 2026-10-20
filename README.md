# Class-balanced LightGBM vs random forest for labelled community membership

Motivation: The paper "Amoc: A multifaceted machine learning-based toolkit for analysing cybercriminal communities on the darknet" (2021) argues that machine learning on network structure can help analyse communities in darknet forums. This experiment asks whether a more carefully built model can do better on a simpler, public proxy task.

Question: Can a class-balanced, CV-tuned LightGBM model with neighbourhood structure features beat a standard random forest on structural features at predicting labelled community membership of nodes in a public communication network?

Dataset: SNAP email-Eu-core graph with its department labels (about 1,000 nodes, about 25,000 directed edges). This is a stand-in, not the paper's own darknet data, so `same_as_paper` is false.

Baseline: a standard random forest on basic per-node structural features (degree, in/out degree, clustering coefficient, PageRank).

Improved model: a class-balanced LightGBM tuned by 5-fold cross-validation on the training nodes, using the same features plus neighbour-degree statistics, PageRank and clustering of neighbours, reciprocity, and k-core number.

Both models use the same stratified 70/30 node split (seed 42), the same evaluation function, and seeds 42, 43 and 44.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and figures/ if available).

## Reference

This project is motivated by the following paper:

> C Chen, C Peersman, M Edwards (2021). *Amoc: A multifaceted machine learning-based toolkit for analysing cybercriminal communities on the darknet*. … Conference on Big. https://ieeexplore.ieee.org/abstract/document/9671906/

Paper author: Prof. Awais Rashid, University of Bristol (awais.rashid@bristol.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
