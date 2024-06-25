if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

epochs=1
batch_size=32
lr=1e-4
loss=mse
weight_decay=1e-4
dataset="sst2"
log_file="training.log"

wbit=8
abit=8
xqtype="smooth_token"
wqtype="smooth_channel"
num_samples=512


save_path="./save/${dataset}/BERT-BASE/${xqtype}_${wqtype}/BERT-BASE_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss_all/"

python3 -W ignore ./bert/sst2.py \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr ${lr} \
    --weight-decay ${weight_decay} \
    --batch_size ${batch_size} \
    --loss_type ${loss} \
    --mixed_prec True \
    --optimizer adam \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --num_samples ${num_samples} \