if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=swin_tiny_patch4_window7_224
epochs=200
batch_size=100
lr=0.1
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="t2c.log"
wbit=8
abit=8
xqtype="smooth_token"
wqtype="smooth_channel"
ttype=ptq

save_path="./save/imagenet/swin_tiny_patch4_window7_224/smooth_token_smooth_channel/swin_tiny_patch4_window7_224_w8_a8_lr1e-4_batch100_cross_entropyloss_all/t2c/"
pre_trained="./save/imagenet/swin_tiny_patch4_window7_224/smooth_token_smooth_channel/swin_tiny_patch4_window7_224_w8_a8_lr1e-4_batch100_cross_entropyloss_all/model_best.pth.tar"

python3 -W ignore ./imagenet/t2c.py \
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
    
