#!/usr/bin/env bash
cd ..
trys=( 1 )
ngfs=( 128,128 )
# ngfs=( 0 )
classes=P,CP,O,A,T
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 10 10 | xargs -n 1 -I {} echo "2^"{} | bc)
lr=0.01
opt=ADAM
# model=LSTMvoting
model=LSTM
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 128 --work_dir works_JMBD4949_loo_1page_${model}/work_${ngf}_numFeat${numfeat}_128epochs_${lr}lr${opt} \
            --layers ${ngf} --batch_size 128 --lr ${lr} --optim ${opt} \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data /data/carabela_segmentacion/JMBD4949_1page_idx/sequence_groups \
            --te_data /data/carabela_segmentacion/JMBD4949_1page_idx/sequence_groups \
            --class_dict work_JMBD4949_loo_1page/tfidf_4949_loo_classes.txt --LOO true \
            --classes ${classes} --model ${model}
        done
    done
done
cd launchers
