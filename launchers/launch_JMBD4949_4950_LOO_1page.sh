#!/usr/bin/env bash
cd ..
trys=( 1 )
# ngfs=( 128,128 )
ngfs=( 0 )
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH,RED
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 2 2 | xargs -n 1 -I {} echo "2^"{} | bc)
#res_10kwords res_10kwords_withoutNormalize res_10kwords_normRWs
path_file_groups=/data/carabela_segmentacion/JMBD4949_4950_1page_idx/groups
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 50 --work_dir works_JMBD4949_4950_loo_1page/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 50 --lr 0.1 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data work_JMBD4949_4950_loo_1page/tfidf_4949_4950_loo.txt \
            --te_data work_JMBD4949_4950_loo_1page/tfidf_4949_4950_loo.txt \
            --class_dict work_JMBD4949_4950_loo_1page/tfidf_4949_4950_loo_classes.txt --LOO true \
            --path_file_groups ${path_file_groups} --classes ${classes}
        done
    done
done
cd launchers