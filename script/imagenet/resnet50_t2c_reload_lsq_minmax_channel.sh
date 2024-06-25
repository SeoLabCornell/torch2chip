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
log_file="reload.log"
wbit=8
abit=8
xqtype="lsq"
wqtype="minmax_channel"

save_path="./save/imagenet/resnet50/ptq/lsq_minmax_channel/resnet50_w8_a8_lr1e-3_batch64_mseloss_layer_trainTrue/t2c/"
pre_trained="./save/imagenet/resnet50/ptq/lsq_minmax_channel/resnet50_w8_a8_lr1e-3_batch64_mseloss_layer_trainTrue/t2c/t2c_model.pth.tar"

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