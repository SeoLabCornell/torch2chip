if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=vit_base
epochs=200
batch_size=128
lr=0.1
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="t2c.log"
wbit=8
abit=8
xqtype="minmax_token"
wqtype="minmax_channel"
ttype=ptq

save_path="/home/jm2787/MLSys24/torch2chip/save/imagenet/vit_base/minmax_token_minmax_channel/vit_base_w8_a8_lr1e-4_batch100_cross_entropyloss_all/t2c/"
pre_trained="/home/jm2787/MLSys24/torch2chip/save/imagenet/vit_base/minmax_token_minmax_channel/vit_base_w8_a8_lr1e-4_batch100_cross_entropyloss_all/model_best.pth.tar"

python3 -W ignore ./imagenet/t2c.py \
    --save_path ${save_path} \
    --model ${model} \
    --resume ${pre_trained} \
    --log_file ${log_file} \
    --fine_tune \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \
    --evaluate \
    --trainer qattn \
    --swl 32 \
    --sfl 26 \
    --export_samples 1 \
    
