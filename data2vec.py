import numpy as np
import pandas as pd
from collections import defaultdict
import random
class data2vec:
    def __init__(self, raw_data='raw.csv', walk_length = 50, near_node=3,
                 use_cols=False, cols=[1], threshold=0, by_value=True, tolerance=3, rho=1):
        self.walk_len = walk_length
        self.near_node = near_node
        self.gene_name = []
        self.gene = self.process_transition_probs(raw_data, use_cols, cols)
        self.context_dict = defaultdict(list)
        self.threshold = threshold
        self.tolerance = tolerance
        self.by_value = by_value
        self.rho = rho
        
    def node_walk(self, num_walks=1):
        walks=[]    
        for column in self.gene.columns:
            '''
            1. select the target column, 
            2. sort values in descending order,
            3. keep the values greater than the threshold
            '''
            col_target = self.gene[column].to_frame()

            col_target = col_target.sort_values(by = column, ascending = False).dropna() #10,6,5,3,2,1            
            col_target = col_target[col_target[column] > self.threshold]

            node_rank = list(col_target.index)
            self.context_dict = defaultdict(list)
            
            self.dict_by_range(col_target,node_rank,column)
            if self.by_value:
                self.dict_by_value(col_target,node_rank,column)

            for _ in range(num_walks):
                for start_node in node_rank:
                    if col_target.at[start_node, column]>1: #only for start node with value greater 10
                        walk = [start_node]#save the order of the genes
                        
                        while len(walk) < self.walk_len:
                            if len(walk)==1:
                                cur =walk[-1]
                                walk.append(np.random.choice(self.context_dict[cur]))
                            else:
                                cur =walk[-1] 
                                pre =walk[-2]
                                walk.append(self.draw_node(pre,cur)) #alias_jump returns the order of nodes 
                        walks.append(walk)

            self.context_dict.clear()
        return walks


    def process_transition_probs(self, expression, use_cols, cols): 
        gene = pd.read_csv(expression, delimiter=',')
        self.gene_name = list(gene[gene.columns[0]]) #save gene names
        
        gene = gene.drop(gene.columns[0],axis=1) #remove gene names
     
        if use_cols: #use specific columns to generate node seqs
            gene = gene.iloc[:, cols]
        gene.columns=['cond_'+str(i) for i in range(1,len(gene.columns)+1)] #rename the columns

        return gene
    
    def dict_by_value(self,col_target,node_rank,column):

        for i in range(len(node_rank)):
            #diff=max(col_target.at[node_rank[i],column]*self.tolerance/100.0,5)
            diff=col_target.at[node_rank[i],column]*self.tolerance/100.0
            lower_bound = col_target.at[node_rank[i],column]-diff
            upper_bound = col_target.at[node_rank[i],column]+diff
                    
            close = col_target[col_target[column]>lower_bound]
            close = close[close[column]<upper_bound]
            self.context_dict[node_rank[i]]+=list(close.index)                  

    def dict_by_range(self,col_target,node_rank,column):
        for i in range(len(node_rank)):
            for j in range(i - self.near_node, i + self.near_node + 1):
                if i == j or j < 0 or j >= len(node_rank):
                    continue
                else:
                    self.context_dict[node_rank[i]].append(node_rank[j])

    def alias_jump(self,pre,cur):
        cur_nei =sorted(self.context_dict[cur])
        pre_nei =self.context_dict[pre]
        unnormalized_prob=[]
        for nei in cur_nei:
            if nei==pre: #case 1
                unnormalized_prob.append(1)
            elif nei in pre_nei: #case 2
                unnormalized_prob.append(self.p)
            else:
                unnormalized_prob.append(self.q)
        norm_const = sum(unnormalized_prob)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_prob]
        #print(normalized_probs)
#        alias_nei = {}
        J,q=alias_setup(normalized_probs)
        
        #return normalized_probs
        return alias_draw(J,q)
        #return alias_setup(normalized_probs)
    def draw_node(self,pre,cur):
        cur_nei =self.context_dict[cur]
        pre_nei =self.context_dict[pre]
        weights=[]
        for nei in cur_nei:
            if nei==pre or nei in pre_nei: #case 1
                weights.append(1)
            # elif nei in pre_nei: #case 2
            #     weights.append(self.p)
            else:
                weights.append(self.rho)     
        
        return random.choices(cur_nei, weights = weights)[0]
        
def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K) #probability of large
	J = np.zeros(K, dtype=np.int) #small bin save the index of large

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob    #proba*N
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0#update the probability of the large bar
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q  #create the proba

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))  #frist draw the bar
	if np.random.rand() < q[kk]:#second draw
	    return kk
	else:
	    return J[kk]        
        
            
        






















