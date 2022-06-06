#!/usr/bin/env bash
cd ..
trys=( 1 )
ngfs=( 0 128,128 128 )
# ngfs=( 0 )
# classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH,other
classes=P,CP,O,A,T,other
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 7 14 | xargs -n 1 -I {} echo "2^"{} | bc)
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
path_data=work_tr50_te49_groups_5classes
# path_data=work_JMBD4949_4950_loo_1page
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 500 --work_dir works_tr50_te49_groups_5classes/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 50 --lr 0.001 --optim RMSprop \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data ${path_data}/tfidf_tr50.txt \
            --te_data ${path_data}/tfidf_te49.txt \
            --class_dict ${path_data}/tfidf_tr50_classes.txt --LOO false \
            --classes ${classes} --do_test true --auto_lr_find true
        done
    done
done
cd launchers
