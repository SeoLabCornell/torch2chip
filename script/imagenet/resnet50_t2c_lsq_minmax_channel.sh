if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=resnet50
epochs=200
batch_size=64
lr=0.1
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="training.log"
wbit=8
abit=8
xqtype="lsq"
wqtype="minmax_channel"
ttype=ptq

save_path="./save/imagenet/resnet50/ptq/lsq_minmax_channel/resnet50_w8_a8_lr1e-3_batch64_mseloss_layer_trainTrue/t2c/"
pre_trained="./save/imagenet/resnet50/ptq/lsq_minmax_channel/resnet50_w8_a8_lr1e-3_batch64_mseloss_layer_trainTrue/model_best.pth.tar"

python3 -W ignore ./imagenet/t2c.py \
    --save_path ${save_path} \
    --model ${model} \
    --resume ${pre_trained} \
    --fine_tune \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \
    --evaluate \
    --trainer ptq \
    --swl 32 \
    --sfl 26 \
    --export_samples 1 \