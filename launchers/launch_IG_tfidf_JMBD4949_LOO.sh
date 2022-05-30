cd ..
path_resultados=work_JMBD4949_loo
mkdir ${path_resultados}
#### Information Gain
data_path=/data/carabela_segmentacion/JMBD_LOO_4949_4950/JMBD_4949_5clases_loo
prob=0.1
classes=P,CP,O,A,T
path_res_IG=${path_resultados}/resultadosIG_4949.txt
python infogain_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res ${path_res_IG}

#### TFIDF
data_path=/data/carabela_segmentacion/JMBD_LOO_4949_4950/JMBD_4949_5clases_loo
path_res_train=${path_resultados}/tfidf_4949_loo.txt
path_res_classes=${path_resultados}/tfidf_4949_loo_classes.txt
python tf_idf_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${path_res_IG} --path_res_classes ${path_res_classes}
cd launchers