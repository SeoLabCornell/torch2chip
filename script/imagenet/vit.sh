if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=vit_base
epochs=50
batch_size=100
lr=1e-4
loss=cross_entropy
weight_decay=1e-4
dataset="imagenet"
log_file="training.log"

wbit=8
abit=8
xqtype="minmax_token"
wqtype="minmax_channel"
num_samples=500
ttype=qattn

save_path="./save/${dataset}/${model}/${xqtype}_${wqtype}/${model}_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss_all/"

python3 -W ignore ./imagenet/vit.py \
    --save_path ${save_path} \
    --model ${model} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr ${lr} \
    --weight-decay ${weight_decay} \
    --batch_size ${batch_size} \
    --loss_type ${loss} \
    --dataset ${dataset} \
    --mixed_prec True \
    --optimizer adam \
    --trainer ${ttype} \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --num_samples ${num_samples} \
    --fine_tune \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \