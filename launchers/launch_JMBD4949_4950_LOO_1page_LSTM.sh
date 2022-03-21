#!/usr/bin/env bash
cd ..
trys=( 1 )
ngfs=( 128,128 )
# ngfs=( 0 )
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH,RED
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 8 11 | xargs -n 1 -I {} echo "2^"{} | bc)
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 128 --work_dir works_JMBD4949_4950_loo_1page_LSTM/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 128 --lr 0.01 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data /data/carabela_segmentacion/JMBD4949_4950_1page_idx/sequence_groups \
            --te_data /data/carabela_segmentacion/JMBD4949_4950_1page_idx/sequence_groups \
            --class_dict work_JMBD4949_4950_loo_1page/tfidf_4949_4950_loo_classes.txt --LOO true \
            --classes ${classes} --model LSTM
        done
    done
done
cd launchers