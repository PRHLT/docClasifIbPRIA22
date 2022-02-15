#!/usr/bin/env bash
cd ..
trys=( 1 )
ngfs=( 16 )
#ngfs=( 16 32,16 128,64,32  )
numfeats=$(seq 14 14 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
#numfeats=$(seq 2 2 | xargs -n 1 -I {} echo "2^"{} | bc)
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
#        for numfeat in "${numfeats[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 50 --work_dir work_1 \
            --layers ${ngf} --batch_size 64 --lr 0.1 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data work_JMBD4949_loo/tfidf_4949_loo.txt \
            --te_data work_JMBD4949_loo/tfidf_4949_loo.txt \
            --class_dict work_JMBD4949_loo/tfidf_4949_loo_classes.txt
        done
    done
done
cd launchers