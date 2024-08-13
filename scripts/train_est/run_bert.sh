device=$1

bash scripts/train_est.sh $device starcoder bert java
bash scripts/train_est.sh $device codegen bert java
bash scripts/train_est.sh $device starcoder bert python
bash scripts/train_est.sh $device codegen bert python
