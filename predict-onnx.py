import onnxruntime, torch
import numpy as np

input_alphabet = {0: 'SOS', 1: 'EOS', 2: 'a', 3: 'c', 4: 'p', 5: 'l', 6: 'o', 7: 'm', 8: 'u', 9: 'b', 10: 'e', 11: 'r', 12: 's', 13: 'g', 14: 'f', 15: 'i', 16: 'd', 17: 'j', 18: 't', 19: 'n', 20: 'z', 21: 'x', 22: 'v', 23: 'k', 24: 'h', 25: 'y', 26: 'w', 27: 'q'}
input_alphabet_lookup = {v: k for k, v in input_alphabet.items()}
output_alphabet = {0: 'SOS', 1: 'EOS', 2: 'e', 3: 'ɪ̯', 4: 'ɪ', 5: 'k', 6: 'æ', 7: 'p', 8: 'ɑː', 9: 'l', 10: 'ə', 11: 'ɒ', 12: 'm', 13: 'j', 14: 'ʊ', 15: 'b', 16: 'ɛ', 17: 'ɑ', 18: 'ɹ', 19: 's', 20: 'iː', 21: 'ɡ', 22: 'f', 23: 'd', 24: 'z', 25: 'd͡ʒ', 26: 't', 27: 'n', 28: 'v', 29: 'i', 30: 'ɔː', 31: 'a', 32: 'h', 33: 'uː', 34: 'ɛː', 35: 'n̩', 36: 'o', 37: 'l̩', 38: 'ɚ', 39: 'w', 40: 'θ', 41: 'u', 42: 'ʃ', 43: 'm̩', 44: 'ɔ', 45: 'ʌ', 46: 'ŋ', 47: 'ʒ', 48: 'ɜː', 49: 'əː', 50: 't͡ʃ', 51: 'aː', 52: 'r', 53: 'eː', 54: 'ʔ', 55: 'æː', 56: 'ɫ', 57: 'ɝ', 58: 'ɜ', 59: 'ð', 60: 'oː', 61: 'ʊ̯', 62: 'ɪː', 63: 'ʍ', 64: 'ɝː'}


print(len(output_alphabet.items()))

ort_session_encoder = onnxruntime.InferenceSession("encoder.model.onnx", providers=["CPUExecutionProvider"])
ort_session_decoder = onnxruntime.InferenceSession("decoder.model.onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
	return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#x = torch.tensor([[ 8, 15, 11,  6,  9,  6, 10, 11,  6,  7,  1]], dtype=torch.int32)
#word = "antibiotic" 
#word = "antidote" 
word = "maturity"
word_x = [input_alphabet_lookup[c] for c in word] + [1]
print(word, word_x)
#x = torch.tensor([[ 8, 15, 11,  6,  9,  6, 10, 11,  6,  7,  1]], dtype=torch.int32)
x = torch.tensor([word_x ], dtype=torch.int32)

print(x)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session_encoder.get_inputs()[0].name: to_numpy(x)}

ort_encoder_outs = ort_session_encoder.run(None, ort_inputs)

print('EC:')
print(' inputs:', ort_session_encoder.get_inputs())
print(ort_session_encoder.get_inputs()[0].name)
print(ort_encoder_outs)
for x in ort_session_encoder.get_outputs():
	print('-> ', x.name)

print('DC:')
print(' inputs:', ort_session_decoder.get_inputs())
print(ort_session_decoder.get_inputs()[0].name)
print(ort_session_decoder.get_inputs()[1].name)
ort_decoder_inputs = {ort_session_decoder.get_inputs()[0].name: ort_encoder_outs[0],
			ort_session_decoder.get_inputs()[1].name: ort_encoder_outs[1]}

decoder_outputs, decoder_hidden, decoder_attn = ort_session_decoder.run(None, ort_decoder_inputs)
for x in ort_session_decoder.get_outputs():
	print('-> ', x.name)

# 1 x steps x alphabet
print(decoder_outputs)
#print('len:',len(decoder_outputs))
#print('len:',len(decoder_outputs[0]))
#print('len:',len(decoder_outputs[0][0]))
outsym = []
out = []
for row in decoder_outputs[0]:
	arg = np.argmax(row)
	outsym.append(arg)
	out.append(output_alphabet[arg])

print(outsym)
print(out)
print('')



# compare ONNX Runtime and PyTorch results
#np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


