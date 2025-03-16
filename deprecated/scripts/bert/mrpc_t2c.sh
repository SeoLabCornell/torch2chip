if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

epochs=1
batch_size=32
lr=1e-4
loss=mse
weight_decay=1e-4
dataset="mrpc"
log_file="training.log"

wbit=8
abit=8
xqtype="minmax_token"
wqtype="minmax_channel"
num_samples=512

pre_trained="./save/mrpc/BERT-BASE/minmax_token_minmax_channel/BERT-BASE_w8_a8_lr1e-4_batch32_mseloss_all/model_best.pth.tar"
save_path="./save/mrpc/BERT-BASE/minmax_token_minmax_channel/BERT-BASE_w8_a8_lr1e-4_batch32_mseloss_all/t2c/"

python3 -W ignore ./bert/mrpc_t2c.py \
    --model "bert" \
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
    --resume ${pre_trained} \
    --export_samples 1 \