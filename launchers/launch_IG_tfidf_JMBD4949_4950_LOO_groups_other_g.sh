cd ..
path_resultados=work_JMBD4949_4950_loo_groups_other_g
mkdir ${path_resultados}
#### Information Gain
data_path=/data/carabela_segmentacion/JMBD4949_4950_idx
prob=0.1
classes=P,CP,O,A,T,V,R,CEN,DP,D,C,TH
path_res_IG=${path_resultados}/resultadosIG_4949_4950.txt
# python infogain_compute.py --data_path ${data_path} --prob $prob --classes $classes --path_res ${path_res_IG} --all_files True
Nwords=8192
C_max=0.2
alpha=0.2
path_res_c=${path_resultados}/res_f_g_Cmax${C_max}_Nwords${Nwords}.txt
path_res_train=${path_resultados}/tfidf_4949_4950_loo.txt
path_res_classes=${path_resultados}/tfidf_4949_4950_loo_classes.txt
# python calculate_distances.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${path_res_IG} --path_res_classes ${path_res_classes} --all_files True --path_res_c ${path_res_c} --C_max ${C_max} --Nwords ${Nwords}
#### TFIDF
python tf_idf_compute_g.py --data_path ${data_path} --prob $prob --classes $classes --path_res_train ${path_res_train} --IG_file ${path_res_IG} --path_res_classes ${path_res_classes} --all_files True  --path_distances ${path_res_c} --alpha ${alpha}
cd launchers