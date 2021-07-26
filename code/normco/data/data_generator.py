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
from gensim.utils import tokenize
from gensim.models.keyedvectors import KeyedVectors

from utils.text_processing import word_tokenize
from utils.text_processing import word_tokenize
from utils.text_processing import clean_text
from utils.text_processing import load_dict_from_vocab_file
from utils.text_processing import tokens_to_ids
from entity_normalization.model.prepare_batch import load_text_batch
from torch.nn import embeddings
import networkx as nx
from networkx import Graph,DiGraph
stop_words = set(stopwords.words('english'))

def getMentions(file, pmid_list=None):
    mentions = []
    with open(file) as f:
        curr_mentions = []
        for l in f:
            if l in '\n':
                if pmid_list is not None and len(curr_mentions) > 0:
                    pmid = curr_mentions[0].split('|')[0]
                    if pmid in pmid_list:
                        mentions.append(curr_mentions)
                elif len(curr_mentions) > 0:       
                    mentions.append(curr_mentions)
                curr_mentions = []
            else:
                curr_mentions.append(l)
        if len(curr_mentions) > 1:
            if pmid_list is not None:
                pmid = curr_mentions[0].split('|')[0]
                if pmid in pmid_list:
                    mentions.append(curr_mentions)
            else:       
                mentions.append(curr_mentions)
    return mentions

def getAbstractTextFromMentions(mentions, asdict=False):
    if asdict:
        abstracts = {}
        for m in mentions:
            pmid = int(m[0].split('|')[0])
            abstracts[pmid] = m[0].split('|')[2] + m[1].split('|')[2]
    else:
        abstracts = []
        for m in mentions:
            abstracts.append(m[0].split('|')[2] + m[1].split('|')[2])
    return abstracts

#Creates the vocabulary and preprocessed initial embeddings files
def createVocabAndEmbeddings(concept_dict, vocabFileName, embeddingInitFileName, conceptInitFileName, pretrained_embeddings, use_unk_concept=True):
    print("CREATING VOCABULARY...\n")
    abstracts = getAbstractTextFromMentions(train_mentions + dev_mentions + test_mentions)
    vocab = set()
    for a in abstracts:
        a = " ".join([t for t in word_tokenize(a) if t not in stop_words])
        tokens = set(word_tokenize(clean_text(a, removePunct=True, lower=False)))
        vocab.update(tokens)
        vocab.update(set([t.lower() for t in tokens]))

    #Uncomment this to use dictionary as well
    for row in concept_dict.values:
        a = " ".join([t for t in word_tokenize(row[0]) if t not in stop_words])
        tokens = set(word_tokenize(clean_text(a, removePunct=True, lower=False)))
        vocab.update(tokens)
        vocab.update(set([t.lower() for t in tokens]))
        if row[7] not in '':
            for syn in row[7].split('|'):
                a = " ".join([t for t in word_tokenize(syn) if t not in stop_words])
                tokens = set(word_tokenize(clean_text(a, removePunct=True, lower=False)))
                vocab.update(tokens)
                vocab.update(set([t.lower() for t in tokens]))

    print("RESOLVING DUPLICATE VOCAB TOKENS...\n")
    #Resolve same word different case
    network = defaultdict(set)
    for v in vocab:
        head_word = v.lower()
        network[head_word].add(v)
        network[head_word].add(v.lower())
        
    duplicates = {}
    for n in network:
        if len(network[n]) > 1:
            duplicates[n] = network[n]

    wv = KeyedVectors.load_word2vec_format(pretrained_embeddings, binary=True)
    mean = np.mean(wv.syn0)
    var = np.var(wv.syn0)
    #vocab = set()
    for d in duplicates:
        vocab = vocab - network[d]
        found = None
        for candidate in list(network[d]):
            if candidate in wv:
                if found is None:
                    found = (candidate, wv.vocab[candidate].count)
                elif wv.vocab[candidate].count > found[1]:
                    found = (candidate, wv.vocab[candidate].count)
        if found is None:
            #just use lowercase version
            vocab.add(d)
        else:
            vocab.add(found[0])

    print("CREATING INITIAL WORD EMBEDDINGS...\n")
    embeddings = []
    final_vocab = []
    #For pad and unknown
    embeddings.append(np.random.normal(loc=mean, scale=np.sqrt(var), size=(wv.vector_size,)))
    embeddings.append(np.random.normal(loc=mean, scale=np.sqrt(var), size=(wv.vector_size,)))
    count = 0
    for v in sorted(vocab):
        if v not in wv:
            count += 1
            embeddings.append(np.random.normal(loc=mean, scale=np.sqrt(var), size=(wv.vector_size,)))
        else:
            embeddings.append(wv[v])

    embeddings = np.asarray(embeddings)
    np.save(embeddingInitFileName, embeddings)

    with open(vocabFileName, 'w', encoding='utf-8') as f:
        f.write('<pad>\n')
        f.write('<unk>\n')
        for k in sorted(vocab):
        #for k in final_vocab:
            f.write(k.lower() + '\n')

    print("CREATING INITIAL CONCEPT EMBEDDINGS...\n")
    vocab_dict = load_dict_from_vocab_file(vocabFileName)
    if use_unk_concept:
        #First is UNK
        concept_init = [np.random.normal(loc=mean, scale=np.sqrt(var), size=(wv.vector_size,))]
        rows = concept_dict.values[1:]
    else:
        concept_init = []
        rows = concept_dict.values

    for row in rows:
        nostops = " ".join([t for t in word_tokenize(row[0]) if t not in stop_words])
        toks = word_tokenize(clean_text(nostops, removePunct=True))
        ids = tokens_to_ids(toks, vocab_dict)
        concept_init.append(np.sum([embeddings[i] for i in ids ], axis=0) )

    concept_init = np.asarray(concept_init)
    np.save(conceptInitFileName, concept_init)

def getData(mentions, concept_dict, ann_type='Disease', abbreviations=None):
    '''
    Get data from PubTator parsed mentions
    '''

    data = []
    max_d = 0
    missing_ids = {}
    if abbreviations is not None:
        tsvin = pd.read_csv(abbreviations, sep='\t', header=None, quotechar='"')
        abbrev_dict = defaultdict(list)
        for row in tsvin.values:
            abbrev_dict[row[0]].append((row[1],row[2]))
            

    for a in mentions:
        text = "".join([a[0].split("|")[2], a[1].split("|")[2]])
        pmid = a[0].split("|")[0]
        curr_mentions = []
        curr_ids = []
        curr_spans = []
        curr_count = 0
        for m in a[2:]:
            fields = m.split('\t')
            if len(fields) < 6 or (ann_type not in fields[4] and fields[4] not in 'Modifier' and fields[4] not in 'CompositeMention'):
                continue
            ids = fields[5].split('|')
            #count += len(ids)
            surface_text = fields[3]
            if abbreviations is not None:
                abbrevs = abbrev_dict[int(pmid)]
                found = False
                while not found:
                    longest = ('', '')
                    for a in abbrevs:
                        if str(a[0]) in surface_text and any(c.isupper() for c in str(a[0])):
                            if len(a[0]) > len(longest[0]):
                                longest = a
                                found = True
                    if found:
                        surface_text = surface_text.replace(longest[0], longest[1])
                        found = False
                    else:
                        found = True
            for d in ids:
                d = d.strip()
                #Get the preferred name
                for id in d.split('+'):
                    
                    gt_values = concept_dict[concept_dict[1].str.contains(id)].values
                    if gt_values.shape[0] == 0:

                        gt_values = concept_dict[concept_dict[2].str.contains(id)].values

                    if gt_values.shape[0] > 0:
                        curr_ids.append(gt_values[0,1])
                        curr_mentions.append(surface_text)
                        curr_spans.append(str(fields[1]) + ' '  + str(fields[2]))
                        curr_count += 1
                    else:
                        curr_ids.append('<unk>')
                        curr_mentions.append(surface_text)
                        curr_spans.append(str(fields[1]) + ' '  + str(fields[2]))
                        curr_count += 1
                        if id != '-1':
                            missing_ids[id] = surface_text
        max_d = max(max_d, curr_count)
        data.append('|'.join(curr_ids) + '\t' + '|'.join(curr_mentions) + '\t' + '|'.join(curr_spans) + '\t' + pmid)
    
    return data, missing_ids

def getTaggerData(taggerFile, abbreviations=None):
    '''
    Creates training data from 
    '''
    if abbreviations is not None:
        tsvin = pd.read_csv(abbreviations, sep='\t', header=None, quotechar="'")
        abbrev_dict = defaultdict(list)
        for row in tsvin.values:
            abbrev_dict[int(row[0])].append((row[1],row[2]))
    mentions = defaultdict(lambda: [[], [], []])
    with open(taggerFile) as f:
        for l in f:
            fields = l.split('|')
            #Get the context
            span = fields[1]
            pmid = fields[0].strip().split('-')[0].strip()

            surface_text = fields[2].strip()
            if len(fields) == 4:
                label = fields[3].strip()
                if label == "UNKNOWN_Disease":
                    label = "<unk>"
                mentions[pmid][2].append(label)
            #Replace abbreviations
            if abbreviations is not None:
                found = False
                abbrevs = abbrev_dict[int(pmid)]
                while not found:
                    longest = ('', '')
                    for a in abbrevs:
                        if str(a[0]) in surface_text and any(c.isupper() for c in str(a[0])):
                            if len(a[0]) > len(longest[0]):
                                longest = a
                                found = True
                    if found:
                        surface_text = surface_text.replace(longest[0], longest[1])
                        found = False
                    else:
                        found = True

            mentions[pmid][0].append(surface_text)
            mentions[pmid][1].append(span)
            
    data = ["|".join(mentions[pmid][2]) + '\t' + "|".join(mentions[pmid][0]) + '\t' + "|".join(mentions[pmid][1]) + '\t' + str(pmid) for pmid in mentions]
    return data

def getDictData(dict_rows, use_context=False, row_filter=None):
    '''
    Creates training data from the concept dictionary
    '''
    dict_data = []
    for row in dict_rows:
        dict_data.append(row[1] + '\t' + row[0] + '\t \t ')
        if row[7] not in '':
            for syn in row[7].split('|'):
                dict_data.append(row[1] + '\t' + syn + '\t \t ')
    return dict_data


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
    def create_vocab(self, concept_dict,
                                 concepts,
                                 mentions,
                                 vocabFileName, embeddingInitFileName, conceptInitFileName,
                                     pretrained_embeddings, use_unk_concept=True):
        print("CREATING VOCABULARY...\n")
        vocab = set()

        for mention in mentions:
            mention = " ".join([t for t in word_tokenize(mention) if t not in stop_words])
            tokens = set(word_tokenize(clean_text(mention, removePunct=True, lower=False)))
            vocab.update(tokens)
            vocab.update(set([t.lower() for t in tokens]))

        # Uncomment this to use dictionary as well
        for concept in concepts:
            concept = " ".join([t for t in word_tokenize(concept) if t not in stop_words])
            tokens = set(word_tokenize(clean_text(concept, removePunct=True, lower=False)))
            vocab.update(tokens)
            vocab.update(set([t.lower() for t in tokens]))
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
        return vocab
    def read_vocab(self,vocab_dir):
        return load_dict_from_vocab_file(vocab_dir)
    #original input for examples:  a list whose elements are [v[1],v[0]]
    # example is of format [mention,related-ids]
    def load_text_batch(self,examples, vocab, maxlen=20, precleaned=False):
        words = []
        lens = []
        ids = []

        for ex in examples:
            w_curr = []

            words_curr, word_len = text_to_batch(ex[0],
                                                 vocab,
                                                 maxlen=maxlen,
                                                 precleaned=precleaned)
            lens.append(word_len)
            words.append(np.asarray(words_curr))
            ids.append(np.asarray(ex[1:]))
        seq_len = len(words)
        return np.asarray(words), np.asarray(lens), np.asarray(ids), seq_len

    def save_data(self,dir,words,lens,ids,seqlens):
        np.savez(dir, words=np.expand_dims(words, 1),
                 lens=np.expand_dims(lens, 1),
                 ids=np.expand_dims(ids, 1), seq_lens=np.expand_dims(seqlens, 1))

    def build_graph(self,tree):
        concept_graph = HierarchyGraph(tree)
        return concept_graph

    def get_related_concept_ls(self,concept_graph,concept2id,pairs,max_depth,max_nodes,search_method):
        res = {}
        for pair in pairs:
            # pair is of format (mention,concept)
            mention = pair[0]
            concept = pair[1]
            id = concept2id[concept]
            context = concept_graph.get_neighbor_idx(id,max_depth,max_nodes,search_method)
            res[mention]=context
        return res

    def


    def gen_data_dict(self,concepts,mentions,coherence_data,vocab,concept2id):
        data_dicts = {}
        for k in concept.keys():
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
                'seq_lens':np.expand_dims(mseqlens,1)

            }
            h_dict = {
                "words": np.expand_dims(hwords, 1),
                "lens": np.expand_dims(hlens, 1),
                "ids": np.expand_dims(hids, 1),
                'seq_lens': np.expand_dims(hseqlens, 1)

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
    def prepare_data(self,concepts,mentions,paired_data,tree,concept2id,max_depth,max_nodes,search_method):
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
            ls_concepts = ls_concepts+concepts[i]
            ls_mentions = ls_mentions + mentions[i]
        vocab = self.create_vocab(concept2id,ls_concepts,ls_mentions,self.args.vocabFileName,None,None,None,True)
        concept_graph = self.build_graph(tree)

        related_concepts = self.get_related_concept_ls(concept_graph,concept2id,paired_data,max_depth,max_nodes,search_methd)
        data_dicts = self.gen_data_dict(concepts,mentions,related_concepts,vocab,concept2id)
        return data_dicts,vocab


















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data and datasets for training and evaluation")
    parser.add_argument('--traindev_file', type=str, help='The location of the raw training/dev dataset in PubTator format', required=True)
    parser.add_argument('--train_pmids', type=str, help='The location of the list of training PMIDs', default=None)
    parser.add_argument('--dev_pmids', type=str, help='The location of the list of dev PMIDs', default=None)
    parser.add_argument('--test_file', type=str, help='The location of the raw test dataset in PubTator format', required=True)
    parser.add_argument('--test_pmids', type=str, help='The location of the list of test PMIDs', default=None)
    parser.add_argument('--concept_dict', type=str, help='The location of the concept dictionary', required=True)
    parser.add_argument('--pretrained_embeddings', type=str, help='The location of the pretrained word embeddings', required=True)
    parser.add_argument('--vocab_file', type=str, help='The name of the vocabulary file', required=True)
    parser.add_argument('--word_embeddings_init_file', type=str, help='The name of the word embeddings init file', required=True)
    parser.add_argument('--concept_embeddings_init_file', type=str, help='The name of the concept embeddings init file', required=True)
    parser.add_argument('--concept_type', type=str, help='The concept type being considered', default='Disease')
    parser.add_argument('--abbreviations_file', type=str, help='The name of the file with listed abbreviations', default=None)
    parser.add_argument('--use_unk_concept', action='store_true', help='Whether or not to use a special concept for "UNKNOWN"', default=False)
    parser.add_argument('--distant_supervision_data', type=str, help='Location of data generated from distant supervision', required=True)
    parser.add_argument('--trainset_mentions_preprocessed_file', type=str, help='Output file for preprocessed trainset mentions', required=True)
    parser.add_argument('--trainset_dictionary_preprocessed_file', type=str, help='Output file for preprocessed dictionary', required=True)
    parser.add_argument('--trainset_coherence_preprocessed_file', type=str, help='Output file for preprocessed trainset coherence data', required=True)
    parser.add_argument('--trainset_distant_preprocessed_file', type=str, help='Output file for preprocessed distantly supervised data', required=True)
    parser.add_argument('--test_data_file', type=str, help='Output file for testset data', required=True)
    parser.add_argument('--dev_data_file', type=str, help='Output file for devset data', required=True)
    parser.add_argument('--tagger_entities_file', type=str, help='Input file containing tagged entities', required=True)
    parser.add_argument('--tagger_data_file', type=str, help='Output file for tagger generated data', required=True)
    parser.add_argument('--tagger_labels_file', type=str, help='Input file containing labels used by TaggerOne', required=True)
    parser.add_argument('--tagger_labels_output', type=str, help='Output file for labels used by TaggerOne', required=True)
    parser.add_argument('--init_embedding', type=bool,default=False, help='if need initial embeddings',
                        required=True)
    
    args = parser.parse_args()
    
    ##################################
    # Load the concept dictionary
    ##################################
    concept_dict = pd.read_csv(args.concept_dict, sep='\t', header=None, comment="#").fillna('')

    ##################################
    # Load all of the mentions
    ##################################
    with open(args.test_pmids) as f:
        test_pmids = set([l.strip() for l in f])
    test_mentions = getMentions(args.test_file, pmid_list=test_pmids)

    with open(args.train_pmids) as f:
        train_pmids = set([l.strip() for l in f])
    train_mentions = getMentions(args.traindev_file, pmid_list=train_pmids)
    
    with open(args.dev_pmids) as f:
        dev_pmids = set([l.strip() for l in f])
    dev_mentions = getMentions(args.traindev_file, pmid_list=dev_pmids)

    ##################################
    # Create vocab and embeddings files
    ##################################
    createVocabAndEmbeddings(concept_dict, args.vocab_file, args.word_embeddings_init_file, 
            args.concept_embeddings_init_file, args.pretrained_embeddings, args.use_unk_concept)

    
    #################################
    # Create training and testing data
    #################################
    print("GENERATING TRAINSET DATA...\n")
    train_data, _ = getData(train_mentions, concept_dict, args.concept_type)
    dict_data = getDictData(concept_dict.values)

    vocab = load_dict_from_vocab_file(args.vocab_file)
    id_dict = {k:i for i,k in enumerate(concept_dict.values[:,1])}
    
    #Dataset mentions
    print("PREPROCESSING MENTIONS...\n")
    mentiondb = []
    for t in train_data:
        fields = t.split('\t')
        for i,m,s in zip(fields[0].split('|'), fields[1].split('|'), fields[2].split('|')):
            mentiondb.append([i,m,s,fields[3]])
    mwords,mlens,mids,mseqlens = load_text_batch([[v[1], v[0]] for v in mentiondb], vocab, id_dict, 20)
    m_usefeatures = np.asarray([r[2] != ' ' for r in mentiondb])
    mseqlens = np.asarray([[1] for i in range(mseqlens)])
    np.savez(args.trainset_mentions_preprocessed_file, words=np.expand_dims(mwords, 1), lens=np.expand_dims(mlens, 1),
                                                                     ids=np.expand_dims(mids, 1), seq_lens=np.expand_dims(mseqlens, 1))
    
    #Dictionary expansion
    print("PREPROCESSING DICTIONARY...\n")
    dictdb = [t.split('\t') for t in dict_data]
    mwords,mlens,mids,mseqlens = load_text_batch([[v[1], v[0]] for v in dictdb], vocab, id_dict, 20)
    m_usefeatures = np.asarray([r[2] != ' ' for r in dictdb])
    mseqlens = np.asarray([[1] for i in range(mseqlens)])
    np.savez(args.trainset_dictionary_preprocessed_file, words=np.expand_dims(mwords, 1), lens=np.expand_dims(mlens, 1), 
                ids=np.expand_dims(mids, 1), seq_lens=np.expand_dims(mseqlens, 1))
    
    #Coherence data
    print("PREPROCESSING COHERENCE DATA...\n")
    coherencedb = [t.split('\t') for t in train_data]
    cwords = []
    clens = []
    cids = []
    cseq = []
    
    for j,c in enumerate(coherencedb):
        w,l,i,s = load_text_batch([[m,i] for m,i in zip(c[1].split('|'), c[0].split('|'))], 
                                  vocab, id_dict, 20)
        cwords.append(w)
        clens.append(l)
        cids.append(i)
        cseq.append(s)
    c_usefeatures = [True] * len(cwords)
    np.savez(args.trainset_coherence_preprocessed_file, words=np.asarray(cwords), lens=np.asarray(clens), 
                ids=np.asarray(cids), seq_lens=np.asarray(cseq))
    
    #Distantly supervised data
    print("PREPROCESSING DISTANTLY SUPERVISED DATA...\n")
    distantdb = pd.read_csv(args.distant_supervision_data, sep='\t').fillna('').values
    distwords = []
    distlens = []
    distids = []
    distseq = []
    dist_usefeatures = c_usefeatures + [False] * len(distwords)
    for c in distantdb:
        w,l,i,s = load_text_batch([[m,i] for m,i in zip(c[1].split('|'), c[0].split('|'))], 
                                  vocab, id_dict, 20, precleaned=True)
        distwords.append(w)
        distlens.append(l)
        distids.append(i)
        distseq.append(s)
        
    distwords = cwords + distwords
    distlens = clens + distlens
    distids = cids + distids
    distseq = cseq + distseq
    np.savez(args.trainset_distant_preprocessed_file, words=np.asarray(distwords), lens=np.asarray(distlens), 
                ids=np.asarray(distids), seq_lens=np.asarray(distseq))

    print("GENERATING TEST DATA...\n")
    test_data, _ = getData(test_mentions, concept_dict, args.concept_type, abbreviations=args.abbreviations_file)
    with open(args.test_data_file, 'w') as f:
        f.write('ids\tmentions\tspans\tpmid\n')
        for t in test_data:
            f.write(t + '\n')
    test_data_w_abbs, _ = getData(test_mentions, concept_dict, args.concept_type)
    with open(args.test_data_file + '_with_abbreviations', 'w') as f:
        f.write('ids\tmentions\tspans\tpmid\n')
        for t in test_data_w_abbs:
            f.write(t + '\n')

    print("GENERATING DEV DATA...\n")
    dev_data, _ = getData(dev_mentions, concept_dict, args.concept_type)
    with open(args.dev_data_file, 'w') as f:
        f.write('ids\tmentions\tspans\tpmid\n')
        for t in dev_data:
            f.write(t + '\n')

    print("GENERATING TAGGER DATA...\n")
    tagger_data = getTaggerData(args.tagger_entities_file, abbreviations=args.abbreviations_file)
    with open(args.tagger_data_file, 'w') as f:
        f.write('ids\tmentions\tspans\tpmid\n')
        for t in tagger_data:
            f.write(t + '\n')
    tagger_data_w_abbs = getTaggerData(args.tagger_entities_file)
    with open(args.tagger_data_file + '_with_abbreviations', 'w') as f:
        f.write('ids\tmentions\tspans\tpmid\n')
        for t in tagger_data_w_abbs:
            f.write(t + '\n')

    print("GETTING TAGGER LABELS...\n")
    tagger_data = getTaggerData(args.tagger_labels_file, abbreviations=args.abbreviations_file)
    with open(args.tagger_labels_output, 'w') as f:
        f.write('ids\tmentions\tspans\tpmid\n')
        for t in tagger_data:
            f.write(t + '\n')
    tagger_data_w_abbs = getTaggerData(args.tagger_labels_file)
    with open(args.tagger_labels_output + '_with_abbreviations', 'w') as f:
        f.write('ids\tmentions\tspans\tpmid\n')
        for t in tagger_data_w_abbs:
            f.write(t + '\n')
