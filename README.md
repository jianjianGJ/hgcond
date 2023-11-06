# Heterogeneous Graph Condensation

**Abstract**—Graph neural networks greatly facilitate data processing in homogeneous and heterogeneous graphs. However, training
GNNs on large-scale graphs poses a significant challenge to computing resources. It is especially prominent on heterogeneous
graphs, which contain multiple types of nodes and edges, and heterogeneous GNNs are also several times more complex than the
ordinary GNNs. Recently, Graph condensation (GCond) is proposed to address the challenge by condensing large-scale
homogeneous graphs into small-scale informative graphs. Its label-based feature initialization and fully-connected design perform well
on homogeneous graphs. While in heterogeneous graphs, label information generally only exists in specific types of nodes, making it
difficult to be applied directly to heterogeneous graphs. In this paper, we propose heterogeneous graph condensation (HGCond).
HGCond uses clustering information instead of label information for feature initialization, and constructs a sparse connection scheme
accordingly. In addition, we found that the simple parameter exploration strategy in GCond leads to insufficient optimization on
heterogeneous graphs. This paper proposes an exploration strategy based on orthogonal parameter sequences to address the
problem. We experimentally demonstrate that the novel feature initialization and parameter exploration strategy is effective.
Experiments show that HGCond significantly outperforms baselines on multiple datasets. On the dataset DBLP, HGCond can
condense DBLP to 0.5% of its original scale to obtain DBLP-0.005. GNNs trained on DBLP-0.005 can retain nearly 99% accuracy
compared to the GNNs trained on full-scale DBLP.

## Our environment configuration
* python          3.8
* torch           1.12.1+cu113
* torch_geometric 2.1.0
* torch_sparse    0.6.15
* torch_scatter   2.0.9
* prettytable     3.4.1
* sklearn         1.1.2
* tqdm            4.64.0
* numpy           1.21.5

## Dataset
| Dataset | #node_type | #node | #edge_type | #edge | #class|
| :----:  | :----: | :----: | :----: |:----: | :----:  | 
| DBLP | 4 | 26128 | 3 | 119783 | 4 |
| IMDB | 3 | 11616 | 2 | 17106 | 3 |
| ACM | 4 | 10942 | 4 | 273936 | 3 |
| AMiner | 3 | 4891819 | 2 | 12518010 | 8 |
| Freebase | 8 | 180098 | 36 | 1531157 | 7 |

The dataset will be downloaded automatically. If the download fails, you can view the source code of `torch_geometric.datasets` and update the url.

## Run
`python hgcond_main.py --dataset dblp --cond 0.001`

```
Dataset:dblp  
Cond-rate:0.001  
Feature initialization:cluster  
Parameter initialization:orth  
Basicmodel:HeteroSGC  

Original dataset infomations:
+-----------------------------------------------------+
|                  Node info of DBLP                  |
+------------+-------+---------+-------+--------------+
|    TYPE    |  NUM  | CHANNEL | CLASS |    SPLIT     |
+------------+-------+---------+-------+--------------+
|   author   |  4057 |   334   |   4   | 400/400/3257 |
|   paper    | 14328 |   4231  |       |              |
|    term    |  7723 |    50   |       |              |
| conference |   20  |    1    |       |              |
+------------+-------+---------+-------+--------------+
+-----------------------------------------------------------------+
|                        Edge info of DBLP                        |
+-------------------------------+-------+--------------+----------+
|              TYPE             |  NUM  |    SHAPE     | SPARSITY |
+-------------------------------+-------+--------------+----------+
|   ('author', 'to', 'paper')   | 19645 | (14328,4057) |  0.03%   |
|   ('paper', 'to', 'author')   | 19645 | (4057,14328) |  0.03%   |
|    ('paper', 'to', 'term')    | 85810 | (7723,14328) |  0.08%   |
| ('paper', 'to', 'conference') | 14328 |  (20,14328)  |  5.00%   |
|    ('term', 'to', 'paper')    | 85810 | (14328,7723) |  0.08%   |
| ('conference', 'to', 'paper') | 14328 |  (14328,20)  |  5.00%   |
+-------------------------------+-------+--------------+----------+
Start condensing ...
Initialization obtained.
Condasention: 100%|███████████████████████████| 128/128 [00:11<00:00, 10.85it/s]
Condensation finished, taking time:13.73s
The condensed graph is saved as ./synthetic_graphs/dblp-0.001.cond
+--------------------------------------------------------------------------+
|                                dblp-0.001                                |
+-------------------------------+------------+------+----------+-----------+
|              Name             |   shape    | mean | sparsity |   labels  |
+-------------------------------+------------+------+----------+-----------+
|             author            |  (4, 334)  | 0.41 |   0.80   | [2 0 1 1] |
|             paper             | (14, 4231) | 0.09 |   0.67   |           |
|              term             |  (7, 50)   | 0.28 |   0.53   |           |
|           conference          |   (1, 1)   | 2.09 |   1.00   |           |
|                               |            |      |          |           |
|   ('author', 'to', 'paper')   |  (14, 4)   | 0.12 |   1.00   |           |
|   ('paper', 'to', 'author')   |  (4, 14)   | 0.12 |   1.00   |           |
|    ('paper', 'to', 'term')    |  (7, 14)   | 0.05 |   1.00   |           |
| ('paper', 'to', 'conference') |  (1, 14)   | 0.22 |   1.00   |           |
|    ('term', 'to', 'paper')    |  (14, 7)   | 0.09 |   1.00   |           |
| ('conference', 'to', 'paper') |  (14, 1)   | 0.26 |   1.00   |           |
+-------------------------------+------------+------+----------+-----------+
Origin graph:242.50Mb  Condensed graph:0.23Mb
Train on the synthetic graph and test on the real graph
Traning: 100%|█████████████████████████████| 1000/1000 [00:09<00:00, 105.42it/s]
Traning: 100%|█████████████████████████████| 1000/1000 [00:09<00:00, 103.82it/s]
Accuracy:71.40+0.95
```

## Result
|Dataset \ cond-rate|0.001|0.005|0.01|0.02|
| :----:  | :----:  |:----:  |:----:  |:----:  |
|DBLP|71.34 ± 2.53| 93.33 ± 0.47 | 93.27 ± 0.59 | 92.96 ± 0.40 |
|IMDB| 45.40 ± 5.47 | 58.05 ± 0.83 | 57.99 ± 0.85 | 57.78 ± 1.34 |
|ACM| 72.33 ± 8.79 | 83.32 ± 11.02 | 87.97 ± 3.08 | 85.70 ± 4.50 |
|Freebase| 52.25 ± 1.04 | 53.64 ± 0.66 | 55.34 ± 1.41 | 54.96 ± 0.84 |

|Dataset \ cond-rate|0.0001|0.0005|0.001|0.0015|
| :----:  | :----:  |:----:  |:----:  |:----:  |
|AMiner|87.03 ± 0.43| 87.05 ± 0.31 | 87.20 ± 0.25 | 87.18 ± 0.15 |
