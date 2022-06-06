cd ..
tr=tr49
te=te50
path_resultados=work_tr49_te50_perPage
mkdir ${path_resultados}
#### Information Gain
data_path=/data/carabela_segmentacion/idxs_JMBD4949/idxs_clasif_per_page/all_classes_noS
prob=0.1
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH
# classes=P,CP,O,A,T
path_res_IG=${path_resultados}/resultadosIG_${tr}.txt
python infogain_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res ${path_res_IG} --all_files True
#### TFIDF
data_path_te=/data/carabela_segmentacion/idxs_JMBD4950/idxs_clasif_per_page/all_classes_noS
# data_path_prod=work_tr49_te50_groups/prod_files_idxs
path_res_classes=${path_resultados}/tfidf_${tr}_classes.txt
path_res_train=${path_resultados}/tfidf_${tr}.txt
path_res_test=${path_resultados}/tfidf_${te}.txt
path_res_prod=${path_resultados}/tfidf_prod_${te}.txt
python tf_idf_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${path_res_IG} --path_res_classes ${path_res_classes} --path_res_test ${path_res_test} --data_path_te ${data_path_te} --all_files True # --path_res_prod ${path_res_prod} --data_path_prod ${data_path_prod} --all_files True
cd launchers
