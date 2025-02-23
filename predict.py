import sys 
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import onnxruntime

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

def loadModelPytorch(path):
	checkpoint = torch.load(path, map_location=torch.device('cpu'))#,weights_only=True)
	input_lang = checkpoint['input_lang']
	output_lang = checkpoint['output_lang']
	hidden_size = checkpoint['hidden_size']

	encoder = EncoderRNN(input_lang.n_words, hidden_size)
	encoder.load_state_dict(checkpoint['encoder'])
	encoder.eval()

	decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)
	decoder.load_state_dict(checkpoint['decoder'])
	decoder.eval()

	return encoder, decoder, input_lang, output_lang

def loadModelONNX(path_encoder, path_decoder):
	encoder = onnxruntime.InferenceSession(path_encoder, providers=["CPUExecutionProvider"])
	decoder = onnxruntime.InferenceSession(path_decoder, providers=["CPUExecutionProvider"])

	return encoder, decoder

def decodePytorch(encoder, decoder, word, input_lang, output_lang):
	sentence = ' '.join([c for c in word])
	chars, attn, encoder_outputs, encoder_hidden = evaluate(encoder, decoder, sentence, input_lang, output_lang, device)
	return chars

def decodeONNX(encoder, decoder, word):
	input_index2char = {0: 'SOS', 1: 'EOS', 2: 'a', 3: 'c', 4: 'p', 5: 'l', 6: 'o', 7: 'm', 8: 'u', 9: 'b', 10: 'e', 11: 'r', 12: 's', 13: 'g', 14: 'f', 15: 'i', 16: 'd', 17: 'j', 18: 't', 19: 'n', 20: 'z', 21: 'x', 22: 'v', 23: 'k', 24: 'h', 25: 'y', 26: 'w', 27: 'q'}
	input_char2index = {v: k for k, v in input_index2char.items()}
	output_index2char = {0: 'SOS', 1: 'EOS', 2: 'e', 3: 'ɪ̯', 4: 'ɪ', 5: 'k', 6: 'æ', 7: 'p', 8: 'ɑː', 9: 'l', 10: 'ə', 11: 'ɒ', 12: 'm', 13: 'j', 14: 'ʊ', 15: 'b', 16: 'ɛ', 17: 'ɑ', 18: 'ɹ', 19: 's', 20: 'iː', 21: 'ɡ', 22: 'f', 23: 'd', 24: 'z', 25: 'd͡ʒ', 26: 't', 27: 'n', 28: 'v', 29: 'i', 30: 'ɔː', 31: 'a', 32: 'h', 33: 'uː', 34: 'ɛː', 35: 'n̩', 36: 'o', 37: 'l̩', 38: 'ɚ', 39: 'w', 40: 'θ', 41: 'u', 42: 'ʃ', 43: 'm̩', 44: 'ɔ', 45: 'ʌ', 46: 'ŋ', 47: 'ʒ', 48: 'ɜː', 49: 'əː', 50: 't͡ʃ', 51: 'aː', 52: 'r', 53: 'eː', 54: 'ʔ', 55: 'æː', 56: 'ɫ', 57: 'ɝ', 58: 'ɜ', 59: 'ð', 60: 'oː', 61: 'ʊ̯', 62: 'ɪː', 63: 'ʍ', 64: 'ɝː'}
	
	def to_numpy(tensor):
		return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

	input_word = [input_char2index[c] for c in word] + [1]
	input_tensor = torch.tensor([input_word], dtype=torch.int32)

	ort_inputs = {encoder.get_inputs()[0].name: to_numpy(input_tensor)}

	ort_encoder_outs = encoder.run(None, ort_inputs)

	# encoder output + hidden state
	ort_decoder_inputs = {decoder.get_inputs()[0].name: ort_encoder_outs[0],
			decoder.get_inputs()[1].name: ort_encoder_outs[1]}

	decoder_outputs, decoder_hidden, decoder_attn = decoder.run(None, ort_decoder_inputs)

	output_string = []
	for row in decoder_outputs[0]:
		arg = np.argmax(row)
		output_string.append(output_index2char[arg])
	
	return output_string


if __name__ == "__main__":
	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = "cpu"

	print(len(sys.argv))

	engine = ""
	if len(sys.argv) == 2:
		encoder, decoder, input_lang, output_lang = loadModelPytorch(sys.argv[1])
		engine = "torch"
	elif len(sys.argv) == 3:
		encoder, decoder = loadModelONNX(sys.argv[1], sys.argv[2])
		engine = "onnx"
	else:
		print('Usage:')
		print(sys.argv[0] + ' <model path>')
		print('               <encoder path> <decoder path>')
		sys.exit(-1)
		
	result = ""
	if engine == "torch":
		torch.no_grad() # We're not going to be training
		result = decodePytorch(encoder, decoder, "maturity", input_lang, output_lang)
	elif engine == "onnx":
		result = decodeONNX(encoder, decoder, "maturity")	

	print(result)
