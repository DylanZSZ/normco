import numpy as np
import random
import os
import argparse
from data_process import construct_graph,data_split

from model import EditDistance_Classifier

#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='The RNN type for coherence',
                            default='GRU', choices=['LSTM', 'GRU'])
    parser.add_argument('--vocab_file', type=str, help='The location of the vocabulary file', required=True)
    parser.add_argument('--embeddings_file', type=str, help='The location of the pretrained embeddings',
                            required=True)
    parser.add_argument('--disease_embeddings_file', type=str, help='The location of pretrained disease embeddings',
                            default=None)
    parser.add_argument('--train_data', type=str, help='The location of the training data', default=None)
    parser.add_argument('--dictionary_data', type=str, help='The location of the dictionary mentions', default=None)
    parser.add_argument('--distant_data', type=str, help='The location of the distantly supervised data',
                            default=None)
    parser.add_argument('--coherence_data', type=str, help='The location of the coherence data', required=True)
    parser.add_argument('--fake_data', type=str, help='The location of synthetic data to train the coherence model',
                            default=None)
    parser.add_argument('--num_epochs', type=int, help='The number of epochs to run', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size for mini batching', default=32)
    parser.add_argument('--sequence_len', type=int, help='The sequence length for phrases', default=20)
    parser.add_argument('--num_neg', type=int, help='The number of negative examples', default=1)
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
    parser.add_argument('--optimizer', type=str, help='Which optimizer to use', default='sgd',
                            choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam'])
    parser.add_argument('--loss', type=str, help='Which loss function to use', default='maxmargin',
                        choices=['maxmargin', 'xent'])
    parser.add_argument('--eval_every', type=int, help='Number of epochs between each evaluation', default=0)
    parser.add_argument('--disease_dict', type=str, help='The location of the disease dictionary', default=None)
    parser.add_argument('--labels_file', type=str, help='Labels file for inline evaluation', default=None)
    parser.add_argument('--banner_tags', type=str, help='Banner tagged documents for inline evaluation',
                            default=None)
    parser.add_argument('--test_features_file', type=str,
                            help='File containing test features when features are used',
                            default=None)
    parser.add_argument('--use_features', action='store_true', help='Whether or not to use hand crafted features',
                            default=False)
    parser.add_argument('--mention_only', action='store_true', help='Whether or not to use mentions only',
                            default=False)
    parser.add_argument('--logfile', type=str, help='File to log evaluation in', default=None)

    args = parser.parse_args()

    setup_seed(0)
    dir = '../data/datasets'
    f=True
    for dir,_,files in os.walk(dir):
        f=False
        for filename in files[:1]:
            #if filename=='envo.obo':continue# envo.obo has a specific problem and I am not sure why.
            concept_list,concept2id,edges,mention_list,synonym_pairs = construct_graph(os.path.join(dir,filename))
            print(filename,'number of synonym pairs',len(synonym_pairs))
            if len(synonym_pairs)<10 or len(synonym_pairs)>10000:continue#modify the two number to control how many datasets will be tested.
            datasets_folds =  data_split(concept_list=concept_list,synonym_pairs=synonym_pairs,is_unseen=True,test_size=0.33)


            for fold,data_fold in enumerate(datasets_folds):
                mentions_train,concepts_train,mentions_test,concepts_test = data_fold
                data = prepare_data(data_fold)
                trainer = NormCoTrainer(args)



                accu1,accu5 = classifier.eval(mentions_test,concepts_test)
                print(filename,'fold--%d,accu1--%f,accu5--%f'%(fold,accu1,accu5))
                break


    
