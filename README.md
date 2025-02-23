# pytorch-seq2seq

Requirements:

```
torch==2.4.0 onnx onnxruntime onnxscript olive 
```

# Training

```
$ python3 seq2seq.py 
```

## Inference

Export model to ONNX:

```
$ python3 export.py model.pth
```

Inference using Pytorch:
```
$ python3 predict.py model.pth
```

Inference using ONNX:
```
$ python3 predict.py encoder_model.onnx decoder_model.onnx
```

## In the browser

```
$ olive auto-opt -m encoder.model.onnx --o encoder_ort --precision int32
$ olive auto-opt -m decoder.model.onnx --o decoder_ort --precision int32
```

This will make two subdirectories ending in `_ort`, the model files for 
browser-based inference are called e.g. `encoder_ort/model.onnx`.

## Acknowledgements

This code is cobbled together from the [seq2seq example](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py) in the Pytorch examples
repository.
