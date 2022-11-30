import numpy as np
import csv

num_nodes = 19717
num_feats = 500
feat_data = np.zeros((num_nodes, num_feats))

node_labels ={}

_labels ={}
idx_list=[]
feat_save=[]
with open("../../pubmed/Pubmed-Diabetes.NODE.paper.tab") as fp:
    fp.readline()#head of the data
    feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}#feature: column
    
    for i, line in enumerate(fp):
        info = line.split("\t")
        idx_list.append(info[0])
        
        
        if not info[1] in _labels:
            _labels[info[1]] =len(_labels)
        node_labels[info[0]] =_labels[info[1]]
        for word_info in info[2:-1]:
            word_info = word_info.split("=")
            feat_data[i,feat_map[word_info[0]]] = float(word_info[1])
        feat_save.append([info[0]]+list(feat_data[i,:])+[_labels[info[1]]])

gi = open('../../pubmed/pubmed_feature_raw.csv','w',newline='')
cw = csv.writer(gi,delimiter=',')
cw.writerows(feat_save)
gi.close()  