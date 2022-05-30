cd ../../
tr=tr49
te=te50
path_resultados=/data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/${tr}
mkdir -p ${path_resultados}
#### Information Gain
data_path=/data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/${tr}/train
echo ${data_path}
prob=0.1
classes=I,M,F
path_res_IG=${path_resultados}/resultadosIG_${tr}.txt
python infogain_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res ${path_res_IG}

#### TFIDF
# data_path=/data/carabela_segmentacion/JMBD_LOO_4949_4950/JMBD_4949_5clases_loo
data_path_te=/data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/${tr}/test
path_res_classes=${path_resultados}/tfidf_${tr}_classes.txt
path_res_train=${path_resultados}/tfidf_${tr}.txt
path_res_test=${path_resultados}/tfidf_${te}.txt
python tf_idf_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${path_res_IG} --path_res_classes ${path_res_classes} --data_path_te ${data_path_te} --path_res_test ${path_res_test}
cd launchers/IMF