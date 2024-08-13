device=$1

bash scripts/train_est.sh $device starcoder lr python
bash scripts/train_est.sh $device starcoder lr java
# bash scripts/train_est.sh $device codegen lr python
# bash scripts/train_est.sh $device codegen lr java
