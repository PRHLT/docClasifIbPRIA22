#!/usr/bin/env bash
cd ..
trys=( 1 )
ngfs=( 0 128 128,128 )
#ngfs=( 16 32,16 128,64,32  )
numfeats=$(seq 3 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
#numfeats=$(seq 2 2 | xargs -n 1 -I {} echo "2^"{} | bc)
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
classes=P,CP,O,A,T,other
path_data=work_JMBD4950_loo_allFiles
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
#        for numfeat in "${numfeats[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 50 --work_dir works_LOO_JMBD4950_allFiles/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 64 --lr 0.1 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data ${path_data}/tfidf_4950_loo.txt \
            --te_data ${path_data}/tfidf_4950_loo.txt \
            --class_dict ${path_data}/tfidf_4950_loo_classes.txt --LOO true --classes ${classes}
        done
    done
done
cd launchers