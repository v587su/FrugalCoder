device=$1

bash scripts/train_est.sh $device starcoder tcqe python
bash scripts/train_est.sh $device starcoder tcqe java
bash scripts/train_est.sh $device codegen tcqe python
bash scripts/train_est.sh $device codegen tcqe java