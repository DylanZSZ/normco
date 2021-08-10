import gc
import time
import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk
import sys
from normco.model.scoring import *
from normco.model.phrase_model import *
from evaluator import *
from normco.utils.text_processing import *

th.manual_seed(7)
np.random.seed(7)

class NormCoTrainer:
    def __init__(self,args):
        self.args = args
        self.model,self.optimizer,self.loss = None,None,None

    def _build_model(self,id_size,vocab_size):
        args = self.args
        output_dim = 128
        sparse = True
        if args.optimizer in 'adam':
            sparse = False

        if args.model in "GRU":
            rnn = nn.GRU
        elif args.model in "LSTM":
            rnn = nn.LSTM

        # Pick the distance function
        margin = np.sqrt(output_dim)
        if args.scoring_type in "euclidean":
            distance_fn = EuclideanDistance()
        if args.scoring_type in "cosine":
            distance_fn = CosineSimilarity(dim=-1)
            margin = args.num_neg - 1
        elif args.scoring_type in "bilinear":
            distance_fn = BilinearMap(output_dim)
            margin = 1.0

        # Create the normalization model
        model = NormalizationModel(id_size,
                                     disease_embeddings_init=None,
                                     phrase_embeddings_init=None,
                                     vocab_size = vocab_size,
                                     distfn=distance_fn,
                                     rnn=rnn, embedding_dim=args.embedding_dim, output_dim=output_dim,
                                     dropout_prob=args.dropout_prob, sparse=sparse,
                                     use_features=args.use_features)

        # Choose the optimizer
        parameters = []
        default_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in 'feature_layer.weight' or name in 'score_layer.weight':
                    default_params.append(param)
                else:
                    parameters.append({'params': param, 'weight_decay': 0.0})
        parameters.append({'params': default_params})

        if args.optimizer in 'sgd':
            optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2reg, momentum=0.9)
        elif args.optimizer in 'rmsprop':
            optimizer = optim.RMSprop(parameters, lr=args.lr, weight_decay=args.l2reg)
        elif args.optimizer in 'adagrad':
            optimizer = optim.Adagrad(parameters, lr=args.lr, weight_decay=args.l2reg)
        elif args.optimizer in 'adadelta':
            optimizer = optim.Adadelta(parameters, lr=args.lr, weight_decay=args.l2reg)
        elif args.optimizer in 'adam':
            optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2reg)

        # Pick the loss function
        if args.loss in 'maxmargin':
            loss = MaxMarginLoss(margin=margin)
        elif args.loss in 'xent':
            loss = CrossEntropyDistanceLoss()

        return model,optimizer,loss

    def train(self,mention_train,coherence_train,mention_valid,coherence_valid,id_size,vocab_size):
        self.model,self.optimizer,self.loss  = self._build_model(id_size,vocab_size)
        mention_train_loader = DataLoader(
            mention_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.threads,
            collate_fn=mention_train.collate
        )
        coherence_train_loader = DataLoader(
            coherence_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.threads,
            collate_fn=coherence_train.collate
        )
        mention_valid_loader = DataLoader(
            mention_valid,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.threads,
            collate_fn=mention_train.collate
        )
        coherence_valid_loader = DataLoader(
            coherence_valid,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.threads,
            collate_fn=coherence_train.collate
        )
        self._train(mention_train_loader,coherence_train_loader,mention_valid_loader,coherence_valid_loader)

    def _train(self,mention_train_loader,coherence_train_loader,mention_valid_loader,coherence_valid_loader,
              log_dir='./tb',eval_data=None,
              logfile=None):

        '''
        Main training loop
        '''
        n_epochs = self.args.num_epochs
        save_every = self.args.save_every
        save_file_name = self.args.save_file_name
        eval_every = self.args.eval_every
        use_coherence = not(self.args.mention_only)
        # Set mode to training
        self.model.train()
        step = 0
        # Keep track of best results for far
        acc_best = (-1, 0.0)
        patience = 15
        # Training loop
        for e in range(n_epochs):
            # Evaluate
            if eval_every > 0 and (e + 1) % eval_every == 0:
                self.model.eval()
                for mb in tqdm(mention_valid_loader, desc='Epoch %d' % e):
                    mb['words'] = Variable(th.stack(mb['words'], 0))
                    mb['lens'] = Variable(th.cat(mb['lens'], 0))
                    mb['ids'] = Variable(th.stack(mb['ids'], 0))
                    mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
                    if self.model.use_features:
                        mb['features'] = Variable(th.stack(mb['features'], 0))
                    # Mention step
                    # Pass through the model
                    mention_scores = self.model(mb, False)
                    # Get sequence length and number of negatives
                    nneg = mention_scores.size()[2]
                    # Get the loss
                    loss = self.loss(mention_scores.view(-1, nneg))

                # Coherence data
                for cb in tqdm(coherence_valid_loader, desc='Epoch %d' % e):
                    cb['words'] = Variable(th.stack(cb['words'], 0))
                    cb['lens'] = Variable(th.cat(cb['lens'], 0))
                    cb['ids'] = Variable(th.stack(cb['ids'], 0))
                    cb['seq_lens'] = Variable(th.cat(cb['seq_lens'], 0))
                    if self.model.use_features:
                        cb['features'] = Variable(th.stack(cb['features'], 0))
                    # Coherence step
                    # Pass through the model
                    scores = self.model(cb, use_coherence)
                    # Get sequence length and number of negatives
                    nneg = scores.size()[2]
                    # Get the loss
                    loss += self.loss(scores.view(-1, nneg))
                print("valid_loss",loss.data)
                gc.collect()
                gc.collect()
                self.model.train()

            for mb in tqdm(mention_train_loader, desc='Epoch %d' % e):
                mb['words'] = Variable(th.stack(mb['words'], 0))
                mb['lens'] = Variable(th.cat(mb['lens'], 0))
                mb['ids'] = Variable(th.stack(mb['ids'], 0))
                mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
                if self.model.use_features:
                    mb['features'] = Variable(th.stack(mb['features'], 0))
                # Mention step
                self.optimizer.zero_grad()
                # Pass through the model
                mention_scores = self.model(mb, False)
                # Get sequence length and number of negatives
                nneg = mention_scores.size()[2]
                # Get the loss
                loss = self.loss(mention_scores.view(-1, nneg))

                loss.backward(retain_graph=True)
                self.optimizer.step()
                step += 1

            # Coherence data
            for cb in tqdm(coherence_train_loader, desc='Epoch %d' % e):
                cb['words'] = Variable(th.stack(cb['words'], 0))
                cb['lens'] = Variable(th.cat(cb['lens'], 0))
                cb['ids'] = Variable(th.stack(cb['ids'], 0))
                cb['seq_lens'] = Variable(th.cat(cb['seq_lens'], 0))
                if self.model.use_features:
                    cb['features'] = Variable(th.stack(cb['features'], 0))
                # Coherence step
                self.optimizer.zero_grad()
                # Pass through the model
                scores = self.model(cb, use_coherence)
                # Get sequence length and number of negatives
                nneg = scores.size()[2]
                # Get the loss
                loss = self.loss(scores.view(-1, nneg))
                loss.backward(retain_graph=True)
                self.optimizer.step()

                step += 1

            gc.collect()
            print(loss.data)
        # Log final best values
        if logfile:
            with open(logfile, 'a') as f:
                f.write("Best accuracy: %f in epoch %d\n" % (acc_best[1], acc_best[0]))

    def evaluate(self,mention_test_data,coherence_test_data):
        mention_test_loader = DataLoader(
            mention_test_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.threads,
            collate_fn=mention_test_data.collate
        )
        coherence_test_loader = DataLoader(
            coherence_test_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.threads,
            collate_fn=coherence_test_data.collate
        )
        predictions = {'without_coherence':[],'with_coherence':[]}
        true_labels = {'without_coherence':[],'with_coherence':[]}
        self.model.eval()
        for mb in tqdm(mention_test_loader):
            mb['words'] = Variable(th.stack(mb['words'], 0))
            mb['lens'] = Variable(th.cat(mb['lens'], 0))
            mb['ids'] = Variable(th.stack(mb['ids'], 0))
            mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
            if self.model.use_features:
                mb['features'] = Variable(th.stack(mb['features'], 0))
            # Mention step
            # Pass through the model
            scores = self.model(mb, False)
            # Get sequence length and number of negatives
            nneg = scores.size()[2]
            # Get the loss
            loss = self.loss(scores.view(-1, nneg))
            predictions['without_coherence'].append(scores.squeeze(0).data.numpy())
            true_labels['without_coherence'].append(mb['ids'].data.numpy())

        for cb in tqdm(coherence_test_loader):
            cb['words'] = Variable(th.stack(cb['words'], 0))
            cb['lens'] = Variable(th.cat(cb['lens'], 0))
            cb['ids'] = Variable(th.stack(cb['ids'], 0))
            cb['seq_lens'] = Variable(th.cat(cb['seq_lens'], 0))
            if self.model.use_features:
                cb['features'] = Variable(th.stack(cb['features'], 0))
            # Coherence step
            # Pass through the model
            scores = self.model(cb, True)
            print(scores.size())
            predictions['with_coherence'].append(scores.data.numpy())
            true_labels['with_coherence'].append(cb['ids'].data.numpy())
            # Get sequence length and number of negatives
            nneg = scores.size()[2]
            # Get the loss
            loss += self.loss(scores.view(-1, nneg))
        print("test_loss", loss.data)
        evaluator = Evaluator()
        correct_total, nsamples = 0,0
        for type in predictions:
            result = np.vstack(predictions[type])
            result = result.reshape(-1,result.shape[-1])
            true_labs = np.vstack(true_labels[type])[:,:,0]
            true_labs = true_labs.reshape(-1,true_labs.shape[-1])
            print("result shape:",result.shape)
            print("labs_shape:",true_labs.shape)
            print(true_labs)
            acc,correct,ntot = evaluator.accu(result,true_labs,5)
            correct_total+=correct
            nsamples+=ntot
            print("result on {} data is {}".format(type,acc))


        print("result on {} data is {}".format("all", correct_total/nsamples))


