#!/usr/bin/env bash
cd ..
trys=( 1 )
ngfs=( 128,128 )
# ngfs=( 0 )
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH,other
# classes=P,CP,O,A,T
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 7 9 | xargs -n 1 -I {} echo "2^"{} | bc)
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
path_data=work_tr49_te50_groups_g
# path_data=work_JMBD4949_4950_loo_1page
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 500 --work_dir works_tr49_te50_groups_g/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 50 --lr 0.001 --optim RMSprop \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data ${path_data}/tfidf_tr49.txt \
            --te_data ${path_data}/tfidf_te50.txt \
            --class_dict ${path_data}/tfidf_tr49_classes.txt --LOO false \
            --classes ${classes} --do_test true --auto_lr_find true --do_prod true --prod_data ${path_data}/tfidf_prod_te50.txt
        done
    done
done
cd launchers
