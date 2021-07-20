import math
import random
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
from smart_open import smart_open
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import entity_normalization.model.scoring as scoring
from utils import text_processing
from entity_normalization.model.prepare_batch import load_text_batch

th.manual_seed(7)
np.random.seed(7)

class PreprocessedDataset(Dataset):
    '''
    Dataset reader (preprocess and transform into tensors)
    '''

    def __init__(self, concept_dict, synonym_file, num_neg, vocab_dict=None, use_features=False):

        # Load the concept, vocab, and character dictionaries
        self.concept_dict = pd.read_csv(concept_dict, sep='\t', header=None, comment='#').fillna('')
        self.nconcepts = self.concept_dict.shape[0]

        # Load the training triples
        self.textdb = np.load(synonym_file)
        self.words = self.textdb['words']
        self.lens = self.textdb['lens']
        self.disease_ids = self.textdb['ids']  # What is disease id and is it the same as id file
        self.seq_lens = self.textdb['seq_lens']
        self.features_flags = False
        if 'usefeatures' in self.textdb:
            self.features_flags = self.textdb['usefeatures']

        self.id_dict = {k: i for i, k in enumerate(self.concept_dict.values[:, 1])}
        if vocab_dict:
            self.id_vocab = {i: k for k, i in vocab_dict.items()}
        if use_features:
            self.concept_feats = self.preprocess_concept_features()

        self.concept_vocab = list(self.id_dict.keys())

        # Various options
        self.num_neg = num_neg
        self.use_features = use_features

    def __len__(self):
        return self.words.shape[0]

    def preprocess_concept_features(self):
        print("Creating feature dict...")
        concept_feats = defaultdict(lambda: [set(), set(), set(), list(), list()])
        for i, row in enumerate(self.concept_dict.values):
            syn_toks = [set(text_processing.conceptTokenize(s)) for s in [row[0]] + row[7].split('|')]
            stem_toks = [set(text_processing.stem_and_lemmatize(list(toks), lemmatize=False)) for toks in syn_toks]
            ctoks = set([s for toks in syn_toks for s in toks])
            concept_feats[i][0] = ctoks
            concept_feats[i][1] = set(text_processing.stem_and_lemmatize(ctoks, lemmatize=False))
            concept_feats[i][2] = set([''.join([t[0] for t in toks]) for toks in syn_toks])
            concept_feats[i][3] = syn_toks
            concept_feats[i][4] = stem_toks
        print("Done!")
        return concept_feats

    def __getitem__(self, i):
        '''
        Retrieve and preprocess a single item
        '''
        words = self.words[i]
        lens = self.lens[i]
        disease_ids = list(self.disease_ids[i])
        seq_len = self.seq_lens[i]
        features_flag = True
        if self.features_flags:
            features_flag = self.features_flags[i]

        for j in range(len(disease_ids)):
            k = 0
            negs = []
            while k < self.num_neg:
                neg = np.random.randint(0, self.nconcepts)
                if neg != disease_ids[j][0]:
                    negs.append(neg)
                    k += 1
            disease_ids[j] = np.concatenate([disease_ids[j], np.asarray(negs)])

        features = []
        if self.use_features:
            if features_flag:
                for i, d in enumerate(disease_ids):
                    curr_features = []
                    toks = set([self.id_vocab[j] for j in words[i]])
                    for id in d:
                        feats = self.concept_feats[id]
                        stems = set(text_processing.stem_and_lemmatize(toks, lemmatize=False))

                        tok_overlap = toks & feats[0]
                        stem_overlap = stems & feats[1]

                        curr_features.append(np.asarray([float(len(tok_overlap) > 0),
                                                         float(len(stem_overlap) > 0),
                                                         max([float(len(toks & ctoks)) for ctoks in feats[3]]) / len(
                                                             toks),
                                                         max([float(len(stems & cstems)) for cstems in feats[4]]) / len(
                                                             stems)
                                                         ]))
                    features.append(np.asarray(curr_features))
            else:
                features = np.zeros((7, len(disease_ids)))

        # Numpy versions
        return th.from_numpy(words), th.from_numpy(lens), th.from_numpy(np.asarray(disease_ids)), th.from_numpy(
            np.asarray([seq_len])), th.from_numpy(np.asarray(features)).type(th.FloatTensor)

    @classmethod
    def collate(cls, batch):
        '''
        Stacks all of the exampeles in a batch, converts to pytorch tensors
        '''
        words, lens, disease_ids, seq_lens, features = zip(*batch)

        return dict(words=words, lens=lens, disease_ids=disease_ids, seq_lens=seq_lens, features=features)




class NormCoDataset(Dataset):
    '''
    Dataset reader (preprocess and transform into tensors)
    '''

    def __init__(self,mentions,concepts):



        # Load the concept, vocab, and character dictionaries
        self.concept_dict = pd.read_csv(concept_dict, sep='\t', header=None, comment='#').fillna('')
        self.nconcepts = self.concept_dict.shape[0]

        # Load the training triples
        self.textdb = np.load(synonym_file)
        self.words = self.textdb['words']
        self.lens = self.textdb['lens']
        self.disease_ids = self.textdb['ids']  # What is disease id and is it the same as id file
        self.seq_lens = self.textdb['seq_lens']
        self.features_flags = False
        if 'usefeatures' in self.textdb:
            self.features_flags = self.textdb['usefeatures']

        self.id_dict = {k: i for i, k in enumerate(self.concept_dict.values[:, 1])}
        if vocab_dict:
            self.id_vocab = {i: k for k, i in vocab_dict.items()}
        if use_features:
            self.concept_feats = self.preprocess_concept_features()

        self.concept_vocab = list(self.id_dict.keys())

        # Various options
        self.num_neg = num_neg
        self.use_features = use_features

    def __len__(self):
        return self.words.shape[0]

    def preprocess_concept_features(self):
        print("Creating feature dict...")
        concept_feats = defaultdict(lambda: [set(), set(), set(), list(), list()])
        for i, row in enumerate(self.concept_dict.values):
            syn_toks = [set(text_processing.conceptTokenize(s)) for s in [row[0]] + row[7].split('|')]
            stem_toks = [set(text_processing.stem_and_lemmatize(list(toks), lemmatize=False)) for toks in syn_toks]
            ctoks = set([s for toks in syn_toks for s in toks])
            concept_feats[i][0] = ctoks
            concept_feats[i][1] = set(text_processing.stem_and_lemmatize(ctoks, lemmatize=False))
            concept_feats[i][2] = set([''.join([t[0] for t in toks]) for toks in syn_toks])
            concept_feats[i][3] = syn_toks
            concept_feats[i][4] = stem_toks
        print("Done!")
        return concept_feats

    def __getitem__(self, i):
        '''
        Retrieve and preprocess a single item
        '''
        words = self.words[i]
        lens = self.lens[i]
        disease_ids = list(self.disease_ids[i])
        seq_len = self.seq_lens[i]
        features_flag = True
        if self.features_flags:
            features_flag = self.features_flags[i]

        for j in range(len(disease_ids)):
            k = 0
            negs = []
            while k < self.num_neg:
                neg = np.random.randint(0, self.nconcepts)
                if neg != disease_ids[j][0]:
                    negs.append(neg)
                    k += 1
            disease_ids[j] = np.concatenate([disease_ids[j], np.asarray(negs)])

        features = []
        if self.use_features:
            if features_flag:
                for i, d in enumerate(disease_ids):
                    curr_features = []
                    toks = set([self.id_vocab[j] for j in words[i]])
                    for id in d:
                        feats = self.concept_feats[id]
                        stems = set(text_processing.stem_and_lemmatize(toks, lemmatize=False))

                        tok_overlap = toks & feats[0]
                        stem_overlap = stems & feats[1]

                        curr_features.append(np.asarray([float(len(tok_overlap) > 0),
                                                         float(len(stem_overlap) > 0),
                                                         max([float(len(toks & ctoks)) for ctoks in feats[3]]) / len(
                                                             toks),
                                                         max([float(len(stems & cstems)) for cstems in feats[4]]) / len(
                                                             stems)
                                                         ]))
                    features.append(np.asarray(curr_features))
            else:
                features = np.zeros((7, len(disease_ids)))

        # Numpy versions
        return th.from_numpy(words), th.from_numpy(lens), th.from_numpy(np.asarray(disease_ids)), th.from_numpy(
            np.asarray([seq_len])), th.from_numpy(np.asarray(features)).type(th.FloatTensor)

    @classmethod
    def collate(cls, batch):
        '''
        Stacks all of the exampeles in a batch, converts to pytorch tensors
        '''
        words, lens, disease_ids, seq_lens, features = zip(*batch)

        return dict(words=words, lens=lens, disease_ids=disease_ids, seq_lens=seq_lens, features=features)


class PreprocessedFakesDataset(Dataset):
    '''
    Dataset reader (preprocess and transform into tensors)
    '''

    def __init__(self, concept_dict, synonym_file, num_neg, vocab_dict=None, use_features=False):

        # Load the concept, vocab, and character dictionaries
        self.concept_dict = pd.read_csv(concept_dict, sep='\t', header=None, comment='#').fillna('')
        self.nconcepts = self.concept_dict.shape[0]

        # Load the training triples
        self.textdb = np.load(synonym_file)
        self.words = self.textdb['words']
        self.lens = self.textdb['lens']
        self.disease_ids = self.textdb['ids']
        self.seq_lens = self.textdb['seq_lens']

        self.id_dict = {k: i for i, k in enumerate(self.concept_dict.values[:, 1])}
        if vocab_dict:
            self.id_vocab = {i: k for k, i in vocab_dict.items()}
        if use_features:
            self.concept_feats = self.preprocess_concept_features()

        self.concept_vocab = list(self.id_dict.keys())

        # Various options
        self.num_neg = num_neg
        self.use_features = use_features

    def __len__(self):
        return len(self.disease_ids)

    def preprocess_concept_features(self):
        print("Creating feature dict...")
        concept_feats = defaultdict(lambda: [set(), set(), set(), list(), list()])
        for i, row in enumerate(self.concept_dict.values):
            syn_toks = [set(text_processing.conceptTokenize(s)) for s in [row[0]] + row[7].split('|')]
            stem_toks = [set(text_processing.stem_and_lemmatize(list(toks), lemmatize=False)) for toks in syn_toks]
            ctoks = set([s for toks in syn_toks for s in toks])
            concept_feats[i][0] = ctoks
            concept_feats[i][1] = set(text_processing.stem_and_lemmatize(ctoks, lemmatize=False))
            concept_feats[i][2] = set([''.join([t[0] for t in toks]) for toks in syn_toks])
            concept_feats[i][3] = syn_toks
            concept_feats[i][4] = stem_toks
        print("Done!")
        return concept_feats

    def __getitem__(self, i):
        '''
        Retrieve and preprocess a single item
        '''
        disease_ids = list(self.disease_ids[i])
        seq_len = self.seq_lens[i]
        # Select words from disease IDs
        words = []
        lens = []
        for i in range(seq_len[0]):
            k = np.random.randint(len(self.words[disease_ids[i][0]]))
            words.append(self.words[disease_ids[i][0]][k])
            lens.append(self.lens[disease_ids[i][0]][k])

        for j in range(len(disease_ids)):
            k = 0
            negs = []
            while k < self.num_neg:
                neg = np.random.randint(0, self.nconcepts)
                if neg != disease_ids[j][0]:
                    negs.append(neg)
                    k += 1
            disease_ids[j] = np.concatenate([disease_ids[j], np.asarray(negs)])

        features = []
        if self.use_features:
            for i, d in enumerate(disease_ids):
                curr_features = []
                toks = set([self.id_vocab[j] for j in words[i]])
                for id in d:
                    feats = self.concept_feats[id]
                    stems = set(text_processing.stem_and_lemmatize(toks, lemmatize=False))

                    tok_overlap = toks & feats[0]
                    stem_overlap = stems & feats[1]

                    curr_features.append(np.asarray([float(len(tok_overlap) > 0),
                                                     float(len(stem_overlap) > 0),
                                                     max([float(len(toks & ctoks)) for ctoks in feats[3]]) / len(toks),
                                                     max([float(len(stems & cstems)) for cstems in feats[4]]) / len(
                                                         stems)
                                                     ]))
                features.append(np.asarray(curr_features))

        # Numpy versions
        return th.from_numpy(np.asarray(words)), th.from_numpy(np.asarray(lens)), th.from_numpy(
            np.asarray(disease_ids)), th.from_numpy(np.asarray([seq_len])), th.from_numpy(np.asarray(features)).type(
            th.FloatTensor)

    @classmethod
    def collate(cls, batch):
        '''
        Stacks all of the exampeles in a batch, converts to pytorch tensors
        '''
        words, lens, disease_ids, seq_lens, features = zip(*batch)
        return dict(words=words, lens=lens, disease_ids=disease_ids, seq_lens=seq_lens, features=features)
