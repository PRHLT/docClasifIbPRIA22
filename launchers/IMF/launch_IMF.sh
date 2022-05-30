#!/usr/bin/env bash
cd ../../
path=data/JMBD4949_4950/IG_TFIDF
tr=tr49
te=te50
trys=( 1 )
ngfs=( 128,128 )
#ngfs=( 16 32,16 128,64,32  )
numfeats=$(seq 10 10 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
#numfeats=$(seq 2 2 | xargs -n 1 -I {} echo "2^"{} | bc)
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
#        for numfeat in "${numfeats[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 50 --work_dir works_IMF/work_${tr}_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 64 --lr 0.1 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data ${path}/${tr}/tfidf_${tr}.txt \
            --te_data ${path}/${tr}/tfidf_${te}.txt \
            --class_dict ${path}/${tr}/tfidf_${tr}_classes.txt --LOO false --do_test true
        done
    done
done
cd launchers/IMF