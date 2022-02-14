data_path=/data/carabela_segmentacion/JMBD_LOO_4949_4950/JMBD_4949_5clases_loo
prob=0.1
classes=P,CP,O,A,T
path_res_train=tfidf_4949_loo.txt
IG_file=resultadosIG_4949.txt
python tf_idf_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${IG_file}