data_path=/data/carabela_segmentacion/idxs_JMBD4949/idxs_clasif/train/
prob=0.1
classes=P,O,T,CP,S,T
path_res=resultadosIG_4949.txt
python infogain_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res ${path_res}