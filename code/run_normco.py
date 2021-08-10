import numpy as np
import random
import os
import argparse
from data_process import load_data,data_split
from normco_trainer import NormCoTrainer
# from edit_distance import EditDistance_Classifier
from normco.data.data_generator import DataGenerator
from normco.data.datasets import PreprocessedDataset
from torch.utils.data import DataLoader
from data_process import *
#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data processing arguments
    parser = argparse.ArgumentParser(description="Generate data and datasets for training and evaluation")
    parser.add_argument('--data_dir',type=str,help='direct to dataset file')
    parser.add_argument('--require_download',type=bool,default=False,help='whether to download data from directory file')
    parser.add_argument('--directory_file',type=str,default=None,help='the location of the directory file based upon which we retrieve data')
    parser.add_argument('--use_unk_concept', action='store_true',
                        help='Whether or not to use a special concept for "UNKNOWN"', default=False)
    parser.add_argument('--init_embedding', type=bool, default=False, help='if need initial embeddings')
    #,max_depth,max_nodes,search_method
    parser.add_argument('--max_depth',type=int,default=2,help='number of maximum depth')
    parser.add_argument('--max_nodes', type=int, default=3, help='number of maximum nodes')
    parser.add_argument('--search_method', type=str, default='bfs', help='algorithm used to traverse the graph to generate context information for each node')
    # training arguments
    parser.add_argument('--model', type=str, help='The RNN type for coherence',
                            default='GRU', choices=['LSTM', 'GRU'])
    parser.add_argument('--num_epochs', type=int, help='The number of epochs to run', default=1)
    parser.add_argument('--batch_size', type=int, help='Batch size for mini batching', default=32)
    parser.add_argument('--sequence_len', type=int, help='The sequence length for phrases', default=20)
    parser.add_argument('--embedding_dim', type=int, help='embedding dimension', default=128)
    parser.add_argument('--num_neg', type=int, help='The number of negative examples', default=20)
    parser.add_argument('--output_dim', type=int, help='The output dimensionality', default=200)
    parser.add_argument('--lr', type=float, help='The starting learning rate', default=0.001)
    parser.add_argument('--l2reg', type=float, help='L2 weight decay', default=0.0)
    parser.add_argument('--dropout_prob', type=float, help='Dropout probability', default=0.0)
    parser.add_argument('--scoring_type', type=str, help='The type of scoring function to use', default="euclidean",
                        choices=['euclidean', 'bilinear', 'cosine'])
    parser.add_argument('--weight_init', type=str, help='Weights file to initialize the model', default=None)
    parser.add_argument('--threads', type=int, help='Number of parallel threads to run', default=1)
    parser.add_argument('--save_every', type=int, help='Number of epochs between each model save', default=1)
    parser.add_argument('--save_file_name', type=str, help='Name of file to save model to', default='model.pth')
    parser.add_argument('--optimizer', type=str, help='Which optimizer to use', default='adam',
                            choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam'])
    # Using SGD gives errors https://github.com/pytorch/pytorch/issues/30402
    parser.add_argument('--loss', type=str, help='Which loss function to use', default='maxmargin',
                        choices=['maxmargin', 'xent'])
    parser.add_argument('--eval_every', type=int, help='Number of epochs between each evaluation', default=2)

    parser.add_argument('--use_features', action='store_true', help='Whether or not to use hand crafted features',
                            default=False)
    parser.add_argument('--mention_only', action='store_true', help='Whether or not to use mentions only',
                            default=False)
    parser.add_argument('--logfile', type=str, help='File to log evaluation in', default=None)

    args = parser.parse_args()
    setup_seed(0)
    if args.require_download:
        assert directory_file is not None
        get_all_data(args.directory_file,args.data_dir)
        print("data downloading done")

    for d,_,files in os.walk(args.data_dir):
        print(files)
        for filename in files:
            print(filename)
            #if filename=='envo.obo':continue# envo.obo has a specific problem and I am not sure why.
            # concept_list,concept2id,edges,mention_list,synonym_pairs = construct_graph(os.path.join(dir,filename))
            #np.array(name_array), np.array(query_id_array), mention2id, edge_index,edges
            concept_array,query2id_array,mention2id,_,edges = load_data(os.path.join(args.data_dir,filename),True)
            print(filename,'number of synonym pairs',len(mention2id))
            # each pair contains queries(mentions) and their ids
            queries_train,queries_valid,queries_test =  data_split(query2id_array,is_unseen=True,test_size=0.33)

            data_generator = DataGenerator(args)
            #def prepare_data(self,paired_data,tree,concept2id,max_depth,max_nodes,search_method):
            mentions = {
                    'train':queries_train,
                    'valid':queries_valid,
                    'test':queries_test

                }
            concept_ids = {
                    'train': [mention2id[i] for i in queries_train],
                    'valid': [mention2id[i] for i in queries_valid],
                    'test':[mention2id[i] for i in queries_test]

                }
            num_neg = args.num_neg
            data_dicts,vocab = data_generator.prepare_data(concept_ids,mentions,query2id_array,edges,mention2id)
                # import dataset from def __init__(self, concept_dict,data_dict, num_neg, vocab_dict=None, use_features=False):

            mention_data_train = PreprocessedDataset(mention2id,data_dicts['train']['mentions'],num_neg,vocab,False)
            coherence_data_train = PreprocessedDataset(mention2id, data_dicts['train']['hierarchy'], num_neg, vocab,
                                                           False)
            mention_data_valid = PreprocessedDataset(mention2id,data_dicts['valid']['mentions'],num_neg,vocab,False)
            coherence_data_valid = PreprocessedDataset(mention2id, data_dicts['valid']['hierarchy'], num_neg, vocab,
                                                           False)
            mention_data_test = PreprocessedDataset(mention2id,data_dicts['test']['mentions'],num_neg,vocab,False)
            coherence_data_test = PreprocessedDataset(mention2id, data_dicts['test']['hierarchy'], num_neg, vocab,
                                                           False)

            trainer = NormCoTrainer(args)
            n_concepts = len(mention2id.keys())
            n_vocab = len(vocab.keys())
            trainer.train(mention_data_train,coherence_data_train,mention_data_valid,coherence_data_valid,n_concepts,n_vocab)
            trainer.evaluate(mention_data_test,coherence_data_test)
