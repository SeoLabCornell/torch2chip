if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=mobilenetv1
epochs=200
batch_size=64
lr=0.1
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="training.log"
wbit=8
abit=8
xqtype="qdrop"
wqtype="minmax_channel"
ttype=ptq

save_path="./save/imagenet/mobilenetv1/ptq/qdrop_minmax_channel/mobilenetv1_w8_a8_lr1e-3_batch64_mseloss_layer_trainTrue/t2c/"
pre_trained="./save/imagenet/mobilenetv1/ptq/qdrop_minmax_channel/mobilenetv1_w8_a8_lr1e-3_batch64_mseloss_layer_trainTrue/model_best.pth.tar"

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
    --export_samples 8 \
    
