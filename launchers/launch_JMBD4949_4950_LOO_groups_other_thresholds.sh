#!/usr/bin/env bash
cd ..
trys=( 1 )
# ngfs=( 128,128 )
ngfs=( 128,128 )
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH,other
#ngfs=( 16 32,16 128,64,32  )
# numfeats=$(seq 4 15 | xargs -n 1 -I {} echo "2^"{} | bc)
#ngfs=( 0 )
numfeats=$(seq 11 11 | xargs -n 1 -I {} echo "2^"{} | bc)
path_all=work_JMBD4949_4950_loo_groups_other_thresholds
path_res=works_JMBD4949_4950_loo_groups_other_thresholds
mkdir ${path_res}
# path=work_JMBD4949_4950_loo_groups_other
probs=( 0.3 )
for prob in "${probs[@]}"; do
    path=${path_all}/work_JMBD4949_4950_loo_groups_other_${prob}
    for try in $trys; do
    for ngf in "${ngfs[@]}"; do
        for numfeat in $numfeats; do
            pres=${path_res}/works_JMBD4949_4950_loo_groups_12classes_other_${prob}
            mkdir pres
            python main.py --epochs 500 --work_dir ${pres}/work_${ngf}_numFeat${numfeat} \
            --layers ${ngf} --batch_size 50 --lr 0.01 --optim ADAM \
            --num_workers 0 --seed ${try}  --num_feats ${numfeat} \
            --tr_data ${path}/tfidf_4949_4950_loo.txt \
            --te_data ${path}/tfidf_4949_4950_loo.txt \
            --class_dict ${path}/tfidf_4949_4950_loo_classes.txt --LOO true \
            --classes ${classes}
        done
    done
done
done
cd launchers