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
# import model.scoring as scoring
# import model.phrase_model as m

from .model.scoring import *
from .model.phrase_model import *
from .eval import load_eval_data
from .eval import dataToTorch
from .eval import accuracy as eval
from .utils.text_processing import *

th.manual_seed(7)
np.random.seed(7)

class NormCoTrainer:
    def __init__(self,args):
        self.args = args
        self.model,self.optimizer,self.loss = self._build_model(args)

    def _build_model(self,args):
        sparse = True
        if args.optimizer in 'adam':
            sparse = False

        if args.model in "GRU":
            rnn = nn.GRU
        elif args.model in "LSTM":
            rnn = nn.LSTM

        embedding_dim = args.output_dim
        output_dim = args.output_dim

        # Pick the distance function
        margin = np.sqrt(output_dim)
        if args.scoring_type in "euclidean":
            distance_fn = scoring.EuclideanDistance()
        if args.scoring_type in "cosine":
            distance_fn = scoring.CosineSimilarity(dim=-1)
            margin = args.num_neg - 1
        elif args.scoring_type in "bilinear":
            distance_fn = scoring.BilinearMap(output_dim)
            margin = 1.0

        # Load concept embeddings initializer
        disease_embs_init = None
        if args.disease_embeddings_file:
            disease_embs_init = np.load(args.disease_embeddings_file)

        # Load initial word embeddings
        embeddings_init = np.load(args.embeddings_file)

        # Create the normalization model
        model = NormalizationModel(len(coherence_data.id_dict.keys()),
                                     disease_embeddings_init=disease_embs_init,
                                     phrase_embeddings_init=embeddings_init,
                                     distfn=distance_fn,
                                     rnn=rnn, embedding_dim=embedding_dim, output_dim=output_dim,
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

        # Load pretrained weights if given
        if args.weight_init:
            model.load_state_dict(th.load(args.weight_init))
        return model,optimizer,loss
    def train(self,mention_train_loader,coherence_train_loader,mention_valid,coherence_valid):
        self._train(self.model,self.optimizer,self.loss,mention_train_loader,coherence_train_loader,mention_valid,coherence_valid,**args)

    def _train(self,model, optimizer, loss_fn,mention_train_loader,coherence_train_loader,mention_valid,coherence_valid,
              log_dir='./tb', n_epochs=100, save_every=1, save_file_name='model.pth', eval_data=None, eval_every=10,
              logfile=None, use_coherence=True):
        '''
        Main training loop
        '''

        # Set mode to training
        model.train()
        step = 0

        # Keep track of best results for far
        acc_best = (-1, 0.0)
        patience = 15
        # Training loop
        for e in range(n_epochs):
            # Evaluate
            if eval_every > 0 and (e + 1) % eval_every == 0:
                model.eval()
                f = None
                if logfile:
                    f = open(logfile, 'a')
                    f.write("Epoch %d\n" % e)
                acc = eval(model, **eval_data, logfile=f)
                if logfile:
                    f.write('\n')
                    f.close()
                if acc >= acc_best[1]:
                    acc_best = (e, acc)
                    th.save(model.state_dict(), save_file_name + "_bestacc_%05d" % (e))
                elif e - acc_best[0] > patience:
                    # Early stopping
                    break

                model.train()
                gc.collect()

            for mb in tqdm(mention_train_loader, desc='Epoch %d' % e):
                mb['words'] = Variable(th.stack(mb['words'], 0))
                mb['lens'] = Variable(th.cat(mb['lens'], 0))
                mb['disease_ids'] = Variable(th.stack(mb['disease_ids'], 0))
                mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
                if model.use_features:
                    mb['features'] = Variable(th.stack(mb['features'], 0))

                # Mention step
                optimizer.zero_grad()
                # Pass through the model
                mention_scores = model(mb, False)
                # Get sequence length and number of negatives
                nneg = mention_scores.size()[2]
                scores = mention_scores
                # Get the loss
                loss = loss_fn(scores.view(-1, nneg))
                loss.backward(retain_graph=True)
                optimizer.step()

                step += 1
            # # Fake data
            # for fb in tqdm(fakes_loader, desc='Epoch %d' % e):
            #
            #     fb['words'] = Variable(th.stack(fb['words'], 0))
            #     fb['lens'] = Variable(th.cat(fb['lens'], 0))
            #     fb['disease_ids'] = Variable(th.stack(fb['disease_ids'], 0))
            #     fb['seq_lens'] = Variable(th.cat(fb['seq_lens'], 0))
            #     if model.use_features:
            #         fb['features'] = Variable(th.stack(fb['features'], 0))
            #     # Coherence step
            #     optimizer.zero_grad()
            #     # Pass through the model
            #     scores = model(fb, use_coherence)
            #     # Get sequence length and number of negatives
            #     nneg = scores.size()[2]
            #     # Get the loss
            #     loss = loss_fn(scores.view(-1, nneg))
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
            #
            #     step += 1

            # Coherence data
            for cb in tqdm(coherence_train_loader, desc='Epoch %d' % e):

                cb['words'] = Variable(th.stack(cb['words'], 0))
                cb['lens'] = Variable(th.cat(cb['lens'], 0))
                cb['disease_ids'] = Variable(th.stack(cb['disease_ids'], 0))
                cb['seq_lens'] = Variable(th.cat(cb['seq_lens'], 0))
                if model.use_features:
                    cb['features'] = Variable(th.stack(cb['features'], 0))
                # Coherence step
                optimizer.zero_grad()
                # Pass through the model
                scores = model(cb, use_coherence)
                # Get sequence length and number of negatives
                nneg = scores.size()[2]
                # Get the loss
                loss = loss_fn(scores.view(-1, nneg))
                loss.backward(retain_graph=True)
                optimizer.step()

                step += 1

            gc.collect()

        # Log final best values
        if logfile:
            with open(logfile, 'a') as f:
                f.write("Best accuracy: %f in epoch %d\n" % (acc_best[1], acc_best[0]))


    #
    # def _train(self,model, optimizer, loss_fn,mention_loader_train, coherence_loader_train,mention_loader_valid,
    #           log_dir='./tb', n_epochs=100, save_every=1, save_file_name='model.pth', eval_data=None, eval_every=10,
    #           logfile=None, use_coherence=True):
    #     '''
    #     Main training loop
    #     '''
    #
    #     # Set mode to training
    #     model.train()
    #     step = 0
    #
    #     # Keep track of best results for far
    #     acc_best = (-1, 0.0)
    #     patience = 15
    #     # Training loop
    #     for e in range(n_epochs):
    #         # Evaluate
    #         if eval_every > 0 and (e + 1) % eval_every == 0:
    #             model.eval()
    #             f = None
    #             if logfile:
    #                 f = open(logfile, 'a')
    #                 f.write("Epoch %d\n" % e)
    #             acc = eval(model, **eval_data, logfile=f)
    #             if logfile:
    #                 f.write('\n')
    #                 f.close()
    #             if acc >= acc_best[1]:
    #                 acc_best = (e, acc)
    #                 th.save(model.state_dict(), save_file_name + "_bestacc_%05d" % (e))
    #             elif e - acc_best[0] > patience:
    #                 # Early stopping
    #                 break
    #
    #             model.train()
    #             gc.collect()
    #
    #         # Epoch loops
    #         # Dictionary data
    #         for mb in tqdm(dict_loader, desc='Epoch %d' % e):
    #             mb['words'] = Variable(th.stack(mb['words'], 0))
    #             mb['lens'] = Variable(th.cat(mb['lens'], 0))
    #             mb['disease_ids'] = Variable(th.stack(mb['disease_ids'], 0))
    #             mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
    #             if model.use_features:
    #                 mb['features'] = Variable(th.stack(mb['features'], 0))
    #
    #             # Mention step
    #             optimizer.zero_grad()
    #             # Pass through the model
    #             mention_scores = model(mb, False)
    #             # Get sequence length and number of negatives
    #             nneg = mention_scores.size()[2]
    #             scores = mention_scores
    #             # Get the loss
    #             loss = loss_fn(scores.view(-1, nneg))
    #             loss.backward(retain_graph=True)
    #             optimizer.step()
    #
    #             step += 1
    #         # Mention data
    #         for mb in tqdm(mention_loader, desc='Epoch %d' % e):
    #             mb['words'] = Variable(th.stack(mb['words'], 0))
    #             mb['lens'] = Variable(th.cat(mb['lens'], 0))
    #             mb['disease_ids'] = Variable(th.stack(mb['disease_ids'], 0))
    #             mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
    #             if model.use_features:
    #                 mb['features'] = Variable(th.stack(mb['features'], 0))
    #
    #             # Mention step
    #             optimizer.zero_grad()
    #             # Pass through the model
    #             mention_scores = model(mb, False)
    #             # Get sequence length and number of negatives
    #             nneg = mention_scores.size()[2]
    #             scores = mention_scores
    #             # Get the loss
    #             loss = loss_fn(scores.view(-1, nneg))
    #             loss.backward(retain_graph=True)
    #             optimizer.step()
    #
    #             step += 1
    #         # Fake data
    #         for fb in tqdm(fakes_loader, desc='Epoch %d' % e):
    #
    #             fb['words'] = Variable(th.stack(fb['words'], 0))
    #             fb['lens'] = Variable(th.cat(fb['lens'], 0))
    #             fb['disease_ids'] = Variable(th.stack(fb['disease_ids'], 0))
    #             fb['seq_lens'] = Variable(th.cat(fb['seq_lens'], 0))
    #             if model.use_features:
    #                 fb['features'] = Variable(th.stack(fb['features'], 0))
    #             # Coherence step
    #             optimizer.zero_grad()
    #             # Pass through the model
    #             scores = model(fb, use_coherence)
    #             # Get sequence length and number of negatives
    #             nneg = scores.size()[2]
    #             # Get the loss
    #             loss = loss_fn(scores.view(-1, nneg))
    #             loss.backward(retain_graph=True)
    #             optimizer.step()
    #
    #             step += 1
    #         # Distantly supervised data
    #         for mb in tqdm(distant_loader, desc='Epoch %d' % e):
    #             mb['words'] = Variable(th.stack(mb['words'], 0))
    #             mb['lens'] = Variable(th.cat(mb['lens'], 0))
    #             mb['disease_ids'] = Variable(th.stack(mb['disease_ids'], 0))
    #             mb['seq_lens'] = Variable(th.cat(mb['seq_lens'], 0))
    #             if model.use_features:
    #                 mb['features'] = Variable(th.stack(mb['features'], 0))
    #
    #             # Mention step
    #             optimizer.zero_grad()
    #             # Pass through the model
    #             mention_scores = model(mb, use_coherence)
    #             # Get sequence length and number of negatives
    #             nneg = mention_scores.size()[2]
    #             scores = mention_scores
    #             # Get the loss
    #             loss = loss_fn(scores.view(-1, nneg))
    #             loss.backward(retain_graph=True)
    #             optimizer.step()
    #
    #             step += 1
    #         # Coherence data
    #         for cb in tqdm(coherence_loader, desc='Epoch %d' % e):
    #
    #             cb['words'] = Variable(th.stack(cb['words'], 0))
    #             cb['lens'] = Variable(th.cat(cb['lens'], 0))
    #             cb['disease_ids'] = Variable(th.stack(cb['disease_ids'], 0))
    #             cb['seq_lens'] = Variable(th.cat(cb['seq_lens'], 0))
    #             if model.use_features:
    #                 cb['features'] = Variable(th.stack(cb['features'], 0))
    #             # Coherence step
    #             optimizer.zero_grad()
    #             # Pass through the model
    #             scores = model(cb, use_coherence)
    #             # Get sequence length and number of negatives
    #             nneg = scores.size()[2]
    #             # Get the loss
    #             loss = loss_fn(scores.view(-1, nneg))
    #             loss.backward(retain_graph=True)
    #             optimizer.step()
    #
    #             step += 1
    #
    #         gc.collect()
    #
    #     # Log final best values
    #     if logfile:
    #         with open(logfile, 'a') as f:
    #             f.write("Best accuracy: %f in epoch %d\n" % (acc_best[1], acc_best[0]))



