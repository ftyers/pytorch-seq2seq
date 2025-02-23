import sys 
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os

from seq2seq import EncoderRNN, AttnDecoderRNN, tensorFromSentence

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
	return decoded_words, decoder_attn, encoder_outputs, encoder_hidden

with open(sys.argv[1], 'rb') as f:
	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = "cpu"
	checkpoint = torch.load(f, map_location=torch.device('cpu'))
	#checkpoint = torch.load(f)
	input_lang = checkpoint['input_lang']
	output_lang = checkpoint['output_lang']
	hidden_size = 256
	encoder = EncoderRNN(input_lang.n_words, hidden_size)
	encoder.load_state_dict(checkpoint['encoder'])
	decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)
	decoder.load_state_dict(checkpoint['decoder'])

	print(input_lang.index2word)
	print(output_lang.index2word)

#	encoder.cuda()
#	decoder.cuda()

	encoder.eval()
	decoder.eval()

	inp = ' '.join([c for c in sys.argv[2]])
	words, attn, encoder_outputs, encoder_hidden = evaluate(encoder, decoder, inp, input_lang, output_lang, device)
	print(inp, words)

	with torch.no_grad():
		input_tensor = tensorFromSentence(input_lang, inp, device)

		print(input_tensor)

		onnx_program = torch.onnx.dynamo_export(encoder, input_tensor)
		onnx_program.save("encoder.model.onnx")

		print(encoder_hidden)
		print(encoder_hidden.shape)
		kwargs = {'encoder_outputs':encoder_outputs, 'encoder_hidden':encoder_hidden}
		#The dummy_input should be a tuple that has the same number of elements as are required by trained_model.forward(). https://github.com/pytorch/pytorch/issues/20009
		onnx_decoder = torch.onnx.dynamo_export(decoder, **kwargs)
		onnx_decoder.save("decoder.model.onnx")
