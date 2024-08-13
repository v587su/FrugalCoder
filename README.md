# Source code of the paper "Don't Complete It! Preventing Unhelpful Code Completion for Productive and Sustainable Neural Code Completion Systems"

### Datasets
Download CSN and COFIC into the `datasets` folder. The datasets are available at [CSN(python)](https://huggingface.co/datasets/code-search-net/code_search_net) and [COFIC](https://drive.google.com/file/d/1Ai0WMYrIGQQLqBC180mzUVDSbpkgO6uD/view)

The HumanAccept dataset is available at [HumanAccept]()

### Run

#### Generate training dataset for estimators
We first generate a dataset for training estimators. The dataset is generated by querying the LCMs with the training dataset.
Please refer to `scripts/gen_score.sh`.
An example for starcoder:
```
bash scripts/gen_score.sh 0 starcoder python
```


### Train estimators
The scripts for training estimators are in the `scripts/train_est` folder.

### Generate estimation
The scripts for evaluating estimators are in `scripts/eval_estimators.sh`

### Scripts for processing the results
compute_flops: energy_analysis.py
compute_results: eval_estimator.py


