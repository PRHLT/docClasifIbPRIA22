cd ..
path_resultados=work_JMBD4949_loo_1page
mkdir ${path_resultados}
#### Information Gain
data_path=/data/carabela_segmentacion/idxs_JMBD4949/idxs_clasif_per_page/all_classes_noS
prob=0.1
classes=P,CP,O,A,T
path_res_IG=${path_resultados}/resultadosIG_4949.txt
python infogain_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res ${path_res_IG}
#### TFIDF
path_res_train=${path_resultados}/tfidf_4949_loo.txt
path_res_classes=${path_resultados}/tfidf_4949_loo_classes.txt
python tf_idf_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${path_res_IG} --path_res_classes ${path_res_classes}
cd launchers