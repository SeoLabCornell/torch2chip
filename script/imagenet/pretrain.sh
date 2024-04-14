export CUDA_VISIBLE_DEVICES=0

model=mobilenetv1
epochs=150
batch_size=256
lr=0.1
weight_decay=1e-4
dataset="imagenet"
log_file="training.log"
loss_type="soft_ce"

save_path="/home/jm2787/MLSys24/T2C/save/${dataset}/${model}/${model}_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss/"

torchrun --nproc_per_node=1 --master_port 48002 ./imagenet/main.py \
    --save_path ${save_path} \
    --model ${model} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr ${lr} \
    --weight-decay ${weight_decay} \
    --batch_size ${batch_size} \
    --loss_type ${loss_type} \
    --dataset ${dataset} \
    --optimizer sgd \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \
    --mixup 0.8 \
    --cutmix 1.0 \