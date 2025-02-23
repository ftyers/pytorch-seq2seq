import onnxruntime, torch
import numpy as np

input_alphabet = {0: 'SOS', 1: 'EOS', 2: "'", 3: 'm', 4: 'u', 5: 'r', 6: 'i', 7: 'c', 8: 'a', 9: 'b', 10: 'o', 11: 't', 12: 's', 13: 'e', 14: 'd', 15: 'n', 16: 'w', 17: 'l', 18: 'v', 19: 'f', 20: 'g', 21: 'p', 22: 'h', 23: 'x', 24: 'j', 25: 'z', 26: 'k', 27: 'y', 28: 'q', 29: 'ú', 30: 'é', 31: 'á', 32: 'ā', 33: 'í', 34: 'ë', 35: 'ó', 36: 'ä', 37: 'ʼ', 38: 'ç', 39: 'č', 40: 'ł', 41: 'ś', 42: 'ț', 43: 'ī', 44: 'å', 45: 'ö', 46: 'ă', 47: 'ñ', 48: 'ę', 49: 'ș', 50: 'â', 51: 'ê', 52: 'ē', 53: 'ř', 54: 'ū', 55: 'ą', 56: 'ï', 57: 'ğ', 58: 'ń', 59: 'è', 60: 'æ', 61: 'ǃ', 62: 'ǁ', 63: 'ò', 64: 'à', 65: 'ʻ', 66: 'ø', 67: 'œ', 68: 'õ', 69: 'ı', 70: 'ň', 71: 'ô', 72: 'ã', 73: 'ŋ', 74: 'ṭ', 75: 'û', 76: 'î', 77: 'ž', 78: 'ų', 79: 'ù', 80: 'ṛ', 81: 'ð', 82: 'š', 83: 'ý', 84: 'ǂ'}
output_alphabet = {0: 'SOS', 1: 'EOS', 2: 'm', 3: 'ɝ', 4: 'ə', 5: 'k', 6: 'ɪ', 7: 'b', 8: 'a', 9: 'ʊ', 10: 't', 11: 'ɒ', 12: 'z', 13: 'ɔ', 14: 'd', 15: 'iː', 16: 'n', 17: 'w', 18: 'uː', 19: 's', 20: 'ɛ', 21: 'l', 22: 'v', 23: 'f', 24: 'ɔː', 25: 'ɹ', 26: 'e', 27: 'ɡ', 28: 'æ', 29: 'm̩', 30: 'ʌ', 31: 'ŋ', 32: 'p', 33: 'j', 34: 'θ', 35: 'ɑː', 36: 'u', 37: 'n̩', 38: 't͡ʃ', 39: 'ɪ̯', 40: 'ɑ', 41: 'd͡ʒ', 42: 'i', 43: 'h', 44: 'ɛː', 45: 'o', 46: 'l̩', 47: 'ɚ', 48: 'ʃ', 49: 'ʒ', 50: 'ɜː', 51: 'əː', 52: 'aː', 53: 'r', 54: 'eː', 55: 'ʔ', 56: 'æː', 57: 'ɫ', 58: 'ɜ', 59: 'ð', 60: 'oː', 61: 'ʊ̯', 62: 'ɪː', 63: 'ʍ', 64: 'ɝː'}

print(len(output_alphabet.items()))

ort_session_encoder = onnxruntime.InferenceSession("encoder.model.onnx", providers=["CPUExecutionProvider"])
ort_session_decoder = onnxruntime.InferenceSession("decoder.model.onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
	return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

x = torch.tensor([[ 8, 15, 11,  6,  9,  6, 10, 11,  6,  7,  1]], dtype=torch.int32)

print(x)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session_encoder.get_inputs()[0].name: to_numpy(x)}

ort_encoder_outs = ort_session_encoder.run(None, ort_inputs)

print('EC:')
print(' inputs:', ort_session_encoder.get_inputs())
print(ort_encoder_outs)

print('DC:')
print(' inputs:', ort_session_decoder.get_inputs())
ort_decoder_inputs = {ort_session_decoder.get_inputs()[0].name: ort_encoder_outs[0],
			ort_session_decoder.get_inputs()[1].name: ort_encoder_outs[1]}

decoder_outputs, decoder_hidden, decoder_attn = ort_session_decoder.run(None, ort_decoder_inputs)

# 1 x steps x alphabet
print(decoder_outputs)
print('len:',len(decoder_outputs))
print('len:',len(decoder_outputs[0]))
print('len:',len(decoder_outputs[0][0]))
for row in decoder_outputs[0]:
	arg = np.argmax(row)
	print(len(row), output_alphabet[arg],arg)
print('')


# compare ONNX Runtime and PyTorch results
#np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


