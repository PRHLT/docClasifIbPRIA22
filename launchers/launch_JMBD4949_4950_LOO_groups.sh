#!/usr/bin/env bash
cd ..
trys=( 1 )
# ngfs=( 128,128 )
ngfs=( 128 )
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 3 14 | xargs -n 1 -I {} echo "2^"{} | bc)
for try in $trys; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            python main.py --epochs 500 --work_dir works_JMBD4949_4950_loo_groups/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 50 --lr 0.01 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data work_JMBD4949_4950_loo_groups/tfidf_4949_4950_loo.txt \
            --te_data work_JMBD4949_4950_loo_groups/tfidf_4949_4950_loo.txt \
            --class_dict work_JMBD4949_4950_loo_groups/tfidf_4949_4950_loo_classes.txt --LOO true \
            --classes ${classes}
        done
    done
done
cd launchers
