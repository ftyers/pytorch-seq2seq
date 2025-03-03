# pytorch-seq2seq

Requirements:

```
torch==2.4.0 onnx onnxruntime onnxscript olive-ai
```

On ehecatl:
`pip install torch==2.4 --index-url https://download.pytorch.org/whl/cu118`

```
python3 -m venv .
source ./bin/activate
```

# Training

```
$ python3 seq2seq.py 
```

## Inference

Export model to ONNX:

```
$ python3 export.py model.checkpoint
```

Inference using Pytorch:
```
$ python3 predict.py model.checkpoint
```

Inference using ONNX:
```
$ python3 predict.py encoder_model.onnx decoder_model.onnx
```

## In the browser

```
$ olive auto-opt -m encoder_model.onnx --o encoder_ort --precision int32
$ olive auto-opt -m decoder_model.onnx --o decoder_ort --precision int32
```

This will make two subdirectories ending in `_ort`, the model files for 
browser-based inference are called e.g. `encoder_ort/model.onnx`.

## Acknowledgements

This code is cobbled together from the [seq2seq example](https://github.com/pytorch/tutorials/blob/main/intermediate_source/seq2seq_translation_tutorial.py) in the Pytorch examples
repository.
