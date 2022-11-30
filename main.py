import argparse
import data2vec
from gensim.models import Word2Vec
import csv 

def parse_args():

    parser = argparse.ArgumentParser(description="Run gene2vec.")
    parser.add_argument('--input', type=str, default='data/sythetic_random_1_5000_4.csv')

    parser.add_argument('--threshold', type=int, default=1)    
    parser.add_argument('--use-cols', type=str, default=False)
    parser.add_argument('--cols', type=list, default=range(4))#[1,2,3,4,5,6,7,8,9,10] [1,2,3,4,5]

    parser.add_argument('--dimensions', type=int, default=128) #vector dimension
    parser.add_argument('--walk_length', type=int, default=80) #node sequence length
    parser.add_argument('--num_walks', type=int, default=2) #the number of node sequences for each starting node

    parser.add_argument('--window_size', type=int, default=5) #word2vec window size
    parser.add_argument('--near_node', type=int, default=2) #avoid generating node sequence with single node
    
    #True: use node values to generate node seqs, False: use orders to generate node seqs
    parser.add_argument('--by_value', type=str, default=True) 
    #corresponds to w in the paper, used to generate context sets
    parser.add_argument('--tolerance', type=int, default=10)#percent
    #unnormalized weights for node seqs generation
    parser.add_argument('--rho', type=float, default=1)
    
    parser.add_argument('--iter', type=int,default=10)
    parser.add_argument('--workers', type=int, default=1)

    return parser.parse_args()


def main(args):
    G = data2vec.data2vec(raw_data = args.input,
                          use_cols = args.use_cols, 
                          cols = args.cols,
                          walk_length = args.walk_length,
                          near_node = args.near_node,
                          threshold = args.threshold,
                          by_value = args.by_value,
                          tolerance = args.tolerance,
                          rho=args.rho)
    
    walks = G.node_walk(num_walks=args.num_walks)
    
    walks = [list(map(str, walk)) for walk in walks]
    #learn_embeddings(walks)
    model = Word2Vec(walks,    #sentences/walkers
                     seed=1,
                     vector_size=args.dimensions,  #vector dimension
                     window=args.window_size,   #window size 
                     min_count=5, #nodes must present at least 3 times
                     sg=1,  #skip-gram
                     workers=args.workers) #1
    

    # save trained model, the node indexes are saved in the trained model
    model.save("trained_node_vectors.model")
    # save the node indexes and the corresponding names
    with open('order2name.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for idx, name in enumerate(G.gene_name):
           writer.writerow([idx, name])
           
if __name__ == "__main__":
    args = parse_args()
    main(args)
    


















