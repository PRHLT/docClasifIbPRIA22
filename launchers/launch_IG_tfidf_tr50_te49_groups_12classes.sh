cd ..
tr=tr50
te=te49
path_resultados=work_tr50_te49_groups_12classes
mkdir ${path_resultados}
#### Information Gain
data_path=/data/carabela_segmentacion/idxs_JMBD4950/idxs_clasif_all_files
prob=0.1
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH
path_res_IG=${path_resultados}/resultadosIG_${tr}.txt
python infogain_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res ${path_res_IG}
#### TFIDF
data_path_te=/data/carabela_segmentacion/idxs_JMBD4949/idxs_clasif_all_files
path_res_classes=${path_resultados}/tfidf_${tr}_classes.txt
path_res_train=${path_resultados}/tfidf_${tr}.txt
path_res_test=${path_resultados}/tfidf_${te}.txt
python tf_idf_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${path_res_IG} --path_res_classes ${path_res_classes} --path_res_test ${path_res_test} --data_path_te ${data_path_te}
cd launchers