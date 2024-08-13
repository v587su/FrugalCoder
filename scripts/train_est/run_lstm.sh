device=$1

bash scripts/train_est.sh $device starcoder lstm python
bash scripts/train_est.sh $device starcoder lstm java
bash scripts/train_est.sh $device codegen lstm python
bash scripts/train_est.sh $device codegen lstm java