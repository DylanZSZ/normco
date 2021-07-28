import argparse
import pandas as pd
import re
import string
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter
# from gensim.utils import tokenize
# from gensim.models.keyedvectors import KeyedVectors

from ..utils.text_processing import word_tokenize
from ..utils.text_processing import word_tokenize
from ..utils.text_processing import clean_text
from ..utils.text_processing import load_dict_from_vocab_file
from ..utils.text_processing import tokens_to_ids
from..model.prepare_batch import load_text_batch
from ..model.reader_utils import text_to_batch
# from torch.nn import embeddings
import networkx as nx
from networkx import Graph,DiGraph
stop_words = set(stopwords.words('english'))


'''
concept2id dict --> to replace concept to disease



'''

class HierarchyGraph:
    def __init__(self,relations):
        self.graph = self._build_graph(relations)
    def _build_graph(self,relations):
        graph = nx.Graph()
        for pair in relations:
            graph.add_edge(*pair)
        return graph
    def get_neighbor_idx(self,target,max_depth=3,max_nodes=10,search_method='bfs'):
        ls_neighbors = []
        if target in self.graph:
            if search_method=='bfs':
                # get the end node for each edge
                ls_neighbors = [i[1] for i in nx.bfs_edges(self.graph,target,max_depth)]
                #prune the list of neighbors
                ls_neighbors = ls_neighbors[:min(max_nodes,len(ls_neighbors))]
            elif search_method=='dfs':
                # get the end node for each edge
                ls_neighbors = [i[1] for i in nx.dfs_edges(self.graph,target,max_depth)]
                #prune the list of neighbors
                ls_neighbors = ls_neighbors[:min(max_nodes,len(ls_neighbors))]

        return ls_neighbors









class DataGenerator:
    def __init__(self,args):
        self.args = args
    # def createVocabAndEmbeddings(self,concept_dict, vocabFileName, embeddingInitFileName, conceptInitFileName,
    #                              pretrained_embeddings, use_unk_concept=True):
    def create_vocab(self,all_data_pairs,pretrained_embeddings=None, use_unk_concept=True):
        print("CREATING VOCABULARY...\n")
        vocab = set()
        for (mention,concept) in all_data_pairs:
            mention = " ".join([t for t in word_tokenize(mention) if t not in stop_words])
            concept = " ".join([t for t in word_tokenize(concept) if t not in stop_words])
            mention_tokens = set(word_tokenize(clean_text(mention, removePunct=True, lower=False)))
            vocab.update(mention_tokens)
            vocab.update(set([t.lower() for t in mention_tokens]))
            concept_tokens = set(word_tokenize(clean_text(concept, removePunct=True, lower=False)))
            vocab.update(concept_tokens)
            vocab.update(set([t.lower() for t in concept_tokens]))

        # for mention in mentions:
        #     mention = " ".join([t for t in word_tokenize(mention) if t not in stop_words])
        #     tokens = set(word_tokenize(clean_text(mention, removePunct=True, lower=False)))
        #     vocab.update(tokens)
        #     vocab.update(set([t.lower() for t in tokens]))
        #
        # # Uncomment this to use dictionary as well
        # for concept in concepts:
        #     concept = " ".join([t for t in word_tokenize(concept) if t not in stop_words])
        #     tokens = set(word_tokenize(clean_text(concept, removePunct=True, lower=False)))
        #     vocab.update(tokens)
        #     vocab.update(set([t.lower() for t in tokens]))
            # if row[7] not in '':
            #     for syn in row[7].split('|'):
            #         a = " ".join([t for t in word_tokenize(syn) if t not in stop_words])
            #         tokens = set(word_tokenize(clean_text(a, removePunct=True, lower=False)))
            #         vocab.update(tokens)
            #         vocab.update(set([t.lower() for t in tokens]))

        print("RESOLVING DUPLICATE VOCAB TOKENS...\n")
        # Resolve same word different case
        network = defaultdict(set)
        for v in vocab:
            head_word = v.lower()
            network[head_word].add(v)
            network[head_word].add(v.lower())

        duplicates = {}
        for n in network:
            if len(network[n]) > 1:
                duplicates[n] = network[n]
        # wv = KeyedVectors.load_word2vec_format(pretrained_embeddings, binary=True)
        # mean = np.mean(wv.syn0)
        # var = np.var(wv.syn0)
        # vocab = set()
        for d in duplicates:
            vocab = vocab - network[d]
            # just use lowercase version
            vocab.add(d)
        vocab = set([k.lower() for k in vocab])
        vocab.update({'<pad>','<unk>'})
        vocab = sorted(vocab)
        vocab_dict = {}
        for i in range(len(vocab)):
            vocab_dict[vocab[i]]=i
        # with open(vocabFileName, 'w', encoding='utf-8') as f:
        #     f.write('<pad>\n')
        #     f.write('<unk>\n')
        #     for k in sorted(vocab):
        #         # for k in final_vocab:
        #         f.write(k.lower() + '\n')
        '''
        The following code was originally used to find duplicates and match case with the dictionary in the
        pretrained embeddings which we may not need. 
        '''
        # for d in duplicates:
        #     vocab = vocab - network[d]
        #     found = None
        #     for candidate in list(network[d]):
        #         if candidate in wv:
        #             if found is None:
        #                 found = (candidate, wv.vocab[candidate].count)
        #             elif wv.vocab[candidate].count > found[1]:
        #                 found = (candidate, wv.vocab[candidate].count)
        #     if found is None:
        #         # just use lowercase version
        #         vocab.add(d)
        #     else:
        #         vocab.add(found[0])

        # if self.args.init_embedding:
        #     print("CREATING INITIAL WORD EMBEDDINGS...\n")
        #     embeddings = []
        #     final_vocab = []
        #     # For pad and unknown
        #     embeddings.append(np.random.normal(loc=mean, scale=np.sqrt(var), size=(wv.vector_size,)))
        #     embeddings.append(np.random.normal(loc=mean, scale=np.sqrt(var), size=(wv.vector_size,)))
        #     count = 0
        #     for v in sorted(vocab):
        #         if v not in wv:
        #             count += 1
        #             embeddings.append(np.random.normal(loc=mean, scale=np.sqrt(var), size=(wv.vector_size,)))
        #         else:
        #             embeddings.append(wv[v])
        #
        #     embeddings = np.asarray(embeddings)
        #     np.save(embeddingInitFileName, embeddings)
        #     print("CREATING INITIAL CONCEPT EMBEDDINGS...\n")
        #     vocab_dict = load_dict_from_vocab_file(vocabFileName)
        #     if use_unk_concept:
        #         # First is UNK
        #         concept_init = [np.random.normal(loc=mean, scale=np.sqrt(var), size=(wv.vector_size,))]
        #         rows = concept_dict.values[1:]
        #     else:
        #         concept_init = []
        #         rows = concept_dict.values
        #
        #     for row in rows:
        #         nostops = " ".join([t for t in word_tokenize(row[0]) if t not in stop_words])
        #         toks = word_tokenize(clean_text(nostops, removePunct=True))
        #         ids = tokens_to_ids(toks, vocab_dict)
        #         concept_init.append(np.sum([embeddings[i] for i in ids], axis=0))
        #
        #     concept_init = np.asarray(concept_init)
        #     np.save(conceptInitFileName, concept_init)
        return vocab_dict
    def read_vocab(self,vocab_dir):
        return load_dict_from_vocab_file(vocab_dir)
    #original input for examples:  a list whose elements are [v[1],v[0]]
    # example is of format [mention,related-ids]

    def save_data(self,dir,words,lens,ids,seqlens):
        np.savez(dir, words=np.expand_dims(words, 1),
                 lens=np.expand_dims(lens, 1),
                 ids=np.expand_dims(ids, 1), seq_lens=np.expand_dims(seqlens, 1))

    def build_graph(self,tree):
        concept_graph = HierarchyGraph(tree)
        return concept_graph

    def get_related_concept_ls(self,concept_graph,concept2id,pairs):
        res = {}
        for pair in pairs:
            # pair is of format (mention,concept)
            mention = pair[0]
            concept = pair[1]
            id = concept2id[concept]
            context = concept_graph.get_neighbor_idx(id,self.args.max_depth,self.args.max_nodes,self.args.search_method)
            res[mention]=context
        return res


    def gen_data_dict(self,concepts,mentions,coherence_data,vocab,concept2id):
        data_dicts = {}
        for k in concepts.keys():
            concept_ls = concepts[k]
            mention_ls = mentions[k]
            #mentions
            mwords, mlens, mids, mseqlens = load_text_batch([ [mention_ls[i],[concept2id[concept_ls[i]]]] for i in range(len(concept_ls))],vocab,20)
            #hierarchy data
            hwords, hlens, hids, hseqlens = load_text_batch([[mention,coherence_data[mention]] for mention in mention_ls], vocab,20)

            m_dict = {
                "words":np.expand_dims(mwords,1),
                "lens":np.expand_dims(mlens,1),
                "ids":np.expand_dims(mids,1),
                'seq_lens':np.expand_dims(np.asarray([[mseqlens]]),1)

            }
            h_dict = {
                "words": np.expand_dims(hwords, 1),
                "lens": np.expand_dims(hlens, 1),
                "ids": np.expand_dims(hids, 1),
                'seq_lens': np.expand_dims(np.asarray([[hseqlens]]), 1)

                    }
            data_dicts[k] = {'mentions':m_dict,'hierarchy':h_dict}
        return data_dicts

        # mwords, mlens, mids, mseqlens = load_text_batch([[v[1], v[0]] for v in data], vocab, id_dict, 20)
        # m_usefeatures = np.asarray([r[2] != ' ' for r in dictdb])
        # mseqlens = np.asarray([[1] for i in range(mseqlens)])
        # np.savez(args.trainset_dictionary_preprocessed_file, words=np.expand_dims(mwords, 1),
        #          lens=np.expand_dims(mlens, 1),
        #          ids=np.expand_dims(mids, 1), seq_lens=np.expand_dims(mseqlens, 1))
    #concepts and mentions:
    # dicts {'train':list,'valid':list,'test':list}
    def prepare_data(self,concepts,mentions,paired_data,tree,concept2id):
        '''
        main method for data generation
        workflow:
        construct dictionary (map between concept and id)
        generate vocabulary
        gen mentions data
        gen dictionary data
        gen coherence data
        '''
        #concept2id used as dict

        #first generate vocabulary
        ls_concepts = []
        ls_mentions = []
        for i in concepts.keys():
            ls_concepts = ls_concepts + concepts[i]
            ls_mentions = ls_mentions + mentions[i]
        vocab = self.create_vocab(paired_data,None,True)
        concept_graph = self.build_graph(tree)
        related_concepts = self.get_related_concept_ls(concept_graph,concept2id,paired_data)
        data_dicts = self.gen_data_dict(concepts,mentions,related_concepts,vocab,concept2id)
        return data_dicts,vocab



