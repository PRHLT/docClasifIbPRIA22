#!/usr/bin/env bash
cd ..
trys=( 1 )
ngfs=( 2 256 512 1024 )
# ngfs=( 0 )
# classes=P,CP,O,A,T
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH,other
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 11 11 | xargs -n 1 -I {} echo "2^"{} | bc)
lr=0.001
opt=ADAM
# model=LSTMvoting
models=( LSTMvoting ) #LSTM
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
for try in $trys; do
for model in "${models[@]}"; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 500 --work_dir works_tr49_te50_perPage_${model}/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 128 --lr ${lr} --optim ${opt} \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data work_tr49_te50_perPage/sequence_groups_tr49 \
            --te_data work_tr49_te50_perPage/sequence_groups_te50 \
            --class_dict work_tr49_te50_perPage/tfidf_tr49_classes.txt --LOO false \
            --classes ${classes} --model ${model} --do_test true \
            --auto_lr_find true #--checkpoint_load works_tr49_te50_perPage_${model}/work_${ngf}_numFeat${numfeat}/checkpoints/carabela_docClasifIbPRIA22/4_aqaxvtw9/checkpoints/epoch=51-step=155.ckpt --no-do_train
        done
    done
done
done
cd launchers
