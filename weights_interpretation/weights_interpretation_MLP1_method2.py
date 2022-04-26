import torch, os
import numpy as np

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def load_dict_class(path):
    res = {}
    number_to_class = {}
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip()
        legajo, c = line.split(" ")
        c = int(c)
        res[legajo] = c
        number_to_class[c] = legajo
    return res, number_to_class

def load_word_order(te_data, num_feats):
    def load(path):
        f = open(path, "r")
        line = f.readlines()[0]
        f.close()
        line = line.strip().split()
        line = line[1:-1]
        return line[:num_feats]
    path_te = te_data
    data = load(path_te)
    return data

def save_to_file(fpath, res, b):
    f = open(fpath, "w")
    f.write("Bias de la clase {}\n".format(b))
    for pal, w in res:
        f.write("{} {}\n".format(pal, w))
    f.close()

def save_to_file_query(fpath, res):
    f = open(fpath, "w")
    res_or = ""
    res_and = ""
    for pal, w in res:
        res_and += "{} && ".format(pal)
        res_or += "{} || ".format(pal)
    f.write("{}\n".format(res_or))
    f.write("{}\n".format(res_and))
    f.close()

def print_weight(fname:str, tensor:np.array):
    f = open(fname, "w")
    for i in tensor:
        i = tensor_to_numpy(i)
        i = [str(x) for x in i]
        j = " ".join(i)
        f.write("{}\n".format(j))
    f.close()

def print_bias(fname:str, tensor:np.array):
    f = open(fname, "w")
    for i in tensor:
        f.write("{}\n".format(i))
    f.close()

print_to_file = True
type_order = "perFileKpagesTrainSeparated"
path_weights = "/data2/jose/projects/carabelaPerPage/works_5page" \
               "/work_ngf_128_try_1_numFeat_2048_order_perFileKpagesTrainSeparated/checkpoints/" \
               "best_undertraincriterion.pth"
#path_te_data = "/data/carabela_segmentacion/idxs_JMBD4949/vector_tfidf_test"
path_tr_data = "/data/carabela/perImages/vector_tfidf_5pages_{}_train".format(type_order)
path_te_data = "/data/carabela/perImages/vector_tfidf_5pages_{}_test".format(type_order)
#path_dict_classes = "/data/carabela_segmentacion/idxs_JMBD4949/vector_tfidf_word_per_file_clasif_train_class_dict_save"
path_dict_classes = "/data/carabela/perImages/vector_tfidf_5pages_train_class_dict_save"
path_save = "/data2/jose/projects/docClasifIbPRIA22/works_WI/weights_analysis_MLP_128_method2"
if not os.path.exists(path_save):
    os.mkdir(path_save)
print("Resumming from model {}".format(path_weights))
checkpoint = torch.load(path_weights)
net_state = checkpoint["net_state"]
print(list(net_state.keys()))
for k, v in net_state.items():
    print(k ,v.shape)

weights_0 = net_state['hidden.weight']
bias_0 = net_state['hidden.bias']
weights_3 = net_state['linear.weight']
bias_3 = net_state['linear.bias']
n_classes = len(bias_3)
n_hidden = len(weights_0)
print("N_Hidden: ",n_hidden)

get_n = 2048
word_order = load_word_order(path_te_data, get_n)
class_dict, number_to_class = load_dict_class(path_dict_classes)
if print_to_file:
    path_save_weights = os.path.join(path_save, "w0")
    if not os.path.exists(path_save_weights):
        os.mkdir(path_save_weights)
    for i, weight_hidden in enumerate(weights_0):
        res = []
        bias = bias_0[i]
        for i_hidden, weight in enumerate(weight_hidden):
            pal = word_order[i_hidden]
            res.append((pal, weight))
        path_save_weights_i = os.path.join(path_save_weights, "u{}".format(i))
        save_to_file(path_save_weights_i, res, bias)

    path_save_weights = os.path.join(path_save, "w1")
    if not os.path.exists(path_save_weights):
        os.mkdir(path_save_weights)
    for c, weight_class in enumerate(weights_3):
        res = []
        bias = bias_3[c]
        for i_hidden, weight in enumerate(weight_class):
            pal = "u{}".format(i_hidden)
            res.append((pal, weight))
        legajo = number_to_class[c]
        #print("{} {}".format(c, legajo))
        path_save_weights_i = os.path.join(path_save_weights, legajo+"_c{}".format(c))
        save_to_file(path_save_weights_i, res, bias)


max_groups = 5
max_n_first = 6
path_save = os.path.join(path_save, "querys")
if not os.path.exists(path_save):
    os.mkdir(path_save)
for num, c_name in number_to_class.items():
    fname = os.path.join(path_save, c_name)
    f = open(fname, "w")
    w1 = tensor_to_numpy(weights_3[num])
    arg_sort_group_n = w1.argsort()[-max_groups:][::-1]
    w1_sorted_n = w1[arg_sort_group_n]
    query = ""
    #print("Tipologia: {} - groups {}".format(c_name, arg_sort_group_n))
    for group in arg_sort_group_n:
        group_hidden = tensor_to_numpy(weights_0[group])
        arg_sort_n = group_hidden.argsort()[-max_n_first:][::-1]
        words_and = []
        for w_pos in arg_sort_n:
            word = word_order[w_pos]
            words_and.append(word)
            if c_name in "FILIP,129,N.13":
                print("group: {} [{}] word {} - {}    pos {}".format(group,w1[group], word, group_hidden[w_pos], w_pos))
        ands = "( {} )".format(" && ".join(words_and))
        query += "{} || ".format(ands)
    #print(query)
    f.write(query)
    f.close()
exit()