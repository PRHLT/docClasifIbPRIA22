import os, glob, shutil

path = "/data/carabela_segmentacion/JMBD4949_4950_idx"
path_dest = "/data/carabela_segmentacion/JMBD4949_4950_idx/prod"
classes = [x.lower() for x in "P,CP,O,A,T,V,R,CEN,DP,D,C,TH,RED".split(",")]
# p cp o a t v r cen dp d c th red 
if not os.path.exists(path_dest):
    os.mkdir(path_dest)

files = glob.glob(os.path.join(path, "*idx"))
for file in files:
    c_file = file.split("/")[-1].split("_")[-1].split(".")[0]
    file_to_move = not any(c == c_file for c in classes)
    if file_to_move:
        print(file,c_file, file_to_move)
        shutil.copy(file, path_dest)