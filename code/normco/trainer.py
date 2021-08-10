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


def get_sorted_top_k(array, top_k=1, axis=-1, reverse=False):
    """
    Returns:
        top_sorted_scores: value
        top_sorted_indexes: index
    """
    if reverse:
        axis_length = array.shape[axis]
        partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                                  range(axis_length - top_k, axis_length), axis)
    else:
        partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)

    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores, top_sorted_indexes


class Evaluator():
    def __init__(self):
        pass

    def accu(self, score_matrix, labels, top_k):
        """
        inputs:
            score_matrix: array-like of shape (n_samples, n_classes), which score_matrix[i][j] indicate the probability of sample i belonging to class j
            labels: array-like of shape(n_samples,)
            top_k : top k accu, mostly k equals to 1 or 5
        """
        scores, preds = get_sorted_top_k(score_matrix, top_k=top_k, reverse=True)  # preds: shape(n_samples,top_k)
        labels = labels.reshape(-1, 1).repeat(top_k, axis=-1)  # repeat at the last dimension
        correctness = labels == preds
        return correctness.sum() / len(labels), correctness.sum(),len(labels)


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

        # Load concept embeddings initializer
        # disease_embs_init = None
        # if args.disease_embeddings_file:
        #     disease_embs_init = np.load(args.disease_embeddings_file)
        #
        # # Load initial word embeddings
        # embeddings_init = np.load(args.embeddings_file)

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

        # Load pretrained weights if given
        # if args.weight_init:
        #     model.load_state_dict(th.load(args.weight_init))
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
              log_dir='./tb', n_epochs=1, save_every=1, save_file_name='model.pth', eval_data=None, eval_every=10,
              logfile=None, use_coherence=True):

        '''
        Main training loop
        '''

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


                # f = None
                # if logfile:
                #     f = open(logfile, 'a')
                #     f.write("Epoch %d\n" % e)
                # acc = eval(self.model, **eval_data, logfile=f)
                # if logfile:
                #     f.write('\n')
                #     f.close()
                # if acc >= acc_best[1]:
                #     acc_best = (e, acc)
                #     th.save(self.model.state_dict(), save_file_name + "_bestacc_%05d" % (e))
                # elif e - acc_best[0] > patience:
                #     # Early stopping
                #     break
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
            predictions['with_coherence'].append(scores.squeeze(0).data.numpy())
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
            print(result)
            print(true_labels[type])
            true_labs = np.vstack(true_labels[type])
            acc,correct,ntot = evaluator.accu(result,true_labs,5)
            correct_total+=correct
            nsamples+=ntot
            print("result on {} data is {}".format(type,acc))


        print("result on {} data is {}".format("all", correct_total/nsamples))






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



