# -*- coding: utf-8 -*-
"""
Adapted from:

NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
*******************************************************************************
**Author**: `Sean Robertson <https://github.com/spro>`_
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <https://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence ``Je ne suis pas le chat noir`` → ``I am not the
# black cat``. Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. ``chat noir`` and ``black cat``. Because of the ``ne/pas``
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, device="cpu"):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.
#
# Training
# ========
#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    #return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)
    # FMT: we have to change torch.long (int64) to torch.int32 here for compatibility with JavaScript
    return torch.tensor(indexes, dtype=torch.int32, device=device).view(1, -1)

def tensorsFromPair(pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('ipa', 'eng', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        print('Epoch',epoch)
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        sys.stdout.flush()


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, sentence, input_lang, output_lang, device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, device, n=10):
#> u n s h u n n a b l e
#= ʌ n ʃ ʌ n ə b ə l
#< ʌ n ʃ ʌ n ə b ə l <EOS>
#
    good = 0
    for i in range(n):
        pair = random.choice(pairs)
        print('?', pair[0])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device)
        output_sentence = ' '.join(output_words)
        if output_sentence == pair[1] + ' <EOS>':
            print('=', pair[1])
            good += 1
        else:
            print('<', output_sentence)
            print('>', pair[1])
        print('')

    print(good/n)


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` which will be used to replace rare words later.
#

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and lowercase
    pairs = [[s.strip().lower() for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 20

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



######################################################################
# Training and Evaluating
# =======================
#

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	input_lang, output_lang, pairs = prepareData('ipa', 'eng', True)
	print(random.choice(pairs))

	hidden_size = 256
	batch_size = 512
	
	input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

	print(input_lang.index2word)
	print(output_lang.index2word)
	
	encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
	decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, device=device).to(device)
	
	train(train_dataloader, encoder, decoder, 100, print_every=2, plot_every=5)
	
	checkpoint = {
	    'encoder':encoder.state_dict(),
	    'decoder':decoder.state_dict(),
            'hidden_size': hidden_size,
	    'input_lang':input_lang, 
	    'output_lang': output_lang
	}
	model_name = 'model.checkpoint'
	with open(model_name,'wb') as f:
	    torch.save(checkpoint, f)
	
	encoder.eval()
	decoder.eval()
	evaluateRandomly(encoder, decoder, device)
	

