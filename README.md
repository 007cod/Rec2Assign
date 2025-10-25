<div align="center">
  <h2><b> Balancing Competition for Fairness-aware Task Assignment in Spatial Crowdsourcing
 </b></h2>
</div>

## Introduction
Rec2Assign maximize profit and the assignment success rate by balancing competition among workers, while simultaneously ensuring fairness.

<p align="center">
<img src="./images/framework.pdf" height = "300" alt="" align=center />
</p>

- Rec2Assign is composed of two main components: (1) Task Recommendation, which aims to recommend an optimal task set for each worker aims to increase the profit. We first compute the Valid Task Set for all workers based on spatiotemporal distribution and introduce a Multi-Stage Probabilistic Recommendation (MSPR) Algorithm, which iteratively recommends tasks based on workers’ willingness for maximizing the overall profit. Next, we apply a Supply-Demand Flow Balancing (SDFB) Algorithm to improve assignment success rate by constructing a Supply-Demand Transfer Graph that identifies the most suitable task regions for each worker to balance competition among workers. (2) Task Assignment, which aims to assign an optimal task to each worker in order to maximize profit while maximizing the consistency of assignment success rates across workers. First, each worker selects a task set based on their willingness from the recommended set. We then employ the Fairness-aware algorithm, which prioritizes tasks assignment to workers with lower assignment success rates while still maintaining high overall profit.

## Requirements
- tqdm==4.67.1
- pandas==2.2.3
- numpy==2.2.3
- matplotlib==3.9.2


## Folder Structure

```tex
└── code-and-data
    ├── data                   # Including DiDi and Yueche datasets
    ├── dataloader             # Codes of preprocessing datasets and calculating metrics
    ├── assignment             # Different assignment and recommendation methods
    ├── utils                  # utils files
    ├── main.py                # This is the main file
    └── README.md              # This document
```

## Datasets
You can access the well pre-processed datasets from [[Github]](https://github.com/Yi107/Dataset-for-PCOM), then place the downloaded contents under the correspond dataset folder such as `./dataset/dd`.

## Quick start
1. Download datasets and place them under `./dataset`
2. Install the required Python packages by running:
  ```
  pip install -r requirements.txt
  ```
3. Start:

```
python main.py
```
