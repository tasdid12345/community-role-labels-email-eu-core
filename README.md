# email-Eu-core department labels: random forest baseline vs CV-tuned LightGBM

## Question
Can a class-balanced, CV-tuned LightGBM model with neighbourhood structure features beat a standard random forest on structural features at predicting labelled community membership of nodes in a public communication network?

## Data
SNAP email-Eu-core graph and department labels (downloaded at runtime from snap.stanford.edu). Department labels stand in for community membership; the original darknet forum data is not public.

## Split
A single stratified 70/30 node-level split with seed 42. Nodes belonging to departments with only one member cannot be stratified, so they are always placed in the training set; all other nodes are split stratified by department.

## Method
- Baseline: random forest (500 trees) on degree, in-degree, out-degree, clustering coefficient and PageRank.
- Improved: class-balanced LightGBM, hyper-parameters chosen by 5-fold CV macro-F1 on training nodes only, using the baseline features plus neighbour-degree statistics, neighbour PageRank and clustering means, reciprocity and k-core number.
- Both models are scored once on the same test nodes, with model seeds 42, 43 and 44.
- Metric: macro-F1 on the held-out test nodes (higher is better).

## Running
```
pip install -r requirements.txt
python analysis.py
```
Outputs `results.json` and `figures/macro_f1_comparison.png`.

## Reference

This project is motivated by the following paper:

> C Chen, C Peersman, M Edwards (2021). *Amoc: A multifaceted machine learning-based toolkit for analysing cybercriminal communities on the darknet*. … Conference on Big. https://ieeexplore.ieee.org/abstract/document/9671906/

Paper author: Prof. Awais Rashid, University of Bristol (awais.rashid@bristol.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
