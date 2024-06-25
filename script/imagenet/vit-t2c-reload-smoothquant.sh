if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=vit_small
batch_size=100
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="reload.log"
wbit=8
abit=8
xqtype="smooth"
wqtype="smooth_channel"
ttype=ptq

save_path="./save/imagenet/vit_small/smooth_smooth_channel/vit_small_w8_a8_lr1e-4_batch100_cross_entropyloss_all/t2c/"
pre_trained="./save/imagenet/vit_small/smooth_smooth_channel/vit_small_w8_a8_lr1e-4_batch100_cross_entropyloss_all/t2c/t2c_model.pth.tar"

python3 -W ignore ./imagenet/reload.py \
    --save_path ${save_path} \
    --model ${model} \
    --batch_size ${batch_size} \
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
    