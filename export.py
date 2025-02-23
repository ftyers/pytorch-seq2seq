import sys 
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os

from seq2seq import EncoderRNN, AttnDecoderRNN, Lang, tensorFromSentence

SOS_token = 0
EOS_token = 1

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
	checkpoint = torch.load(f, map_location=torch.device('cpu'))#,weights_only=True)
	#checkpoint = torch.load(f)
	input_lang = checkpoint['input_lang']
	output_lang = checkpoint['output_lang']
	hidden_size = checkpoint['hidden_size']
	encoder = EncoderRNN(input_lang.n_words, hidden_size)
	encoder.load_state_dict(checkpoint['encoder'])
	decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)
	decoder.load_state_dict(checkpoint['decoder'])

	encoder.eval()
	decoder.eval()

	word = ' '.join([c for c in "maturity"])
	chars, attn, encoder_outputs, encoder_hidden = evaluate(encoder, decoder, word, input_lang, output_lang, device)

	with torch.no_grad():
		print('=======================================================================================')
		input_tensor = tensorFromSentence(input_lang, word, device)

		onnx_program = torch.onnx.export(encoder, 
							input_tensor,
							dynamo=True
						)
		onnx_program.save("encoder.model.onnx")

		# The dummy_input should be a tuple that has the same number of elements as are required by 
		# trained_model.forward(). https://github.com/pytorch/pytorch/issues/20009
		onnx_decoder = torch.onnx.export(decoder, (encoder_outputs,encoder_hidden),dynamo=True)#**kwargs)
		onnx_decoder.save("decoder.model.onnx")
