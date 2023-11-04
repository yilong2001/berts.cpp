# berts.cpp

[ggml](https://github.com/ggerganov/ggml) inference of bert family models (bert, distilbert, roberta ...), classification & seq2seq and more.
High quality bert inference in pure C++.

## Description
The main goal of `berts.cpp` is to run the BERT model with simple binary on CPU

* Plain C/C++ implementation without dependencies
* Inherit support for various architectures from ggml (x86 with AVX2, ARM, etc.)
* Choose model size from 32/16 bits per model weigth
* Simple main for using
* CPP rest server
* Benchmarks to validate correctness and speed of inference

## Limitations & TODO
* bert seq2seq
* bard 
* xlnet
* gpt2
* ...

## Usage

### Checkout the ggml submodule
```sh
git submodule update --init --recursive
```
### Download models
Bert sequence classification model provided as a example. 
You can download with the following cmd or directly from huggingface [https://huggingface.co/yilong2001/bert_cls_example].

```sh
pip3 install -r requirements.txt
python3 models/download-ggml.py download bert-base-uncased f32
```

### Install External Library
To build the library or binary, need install external library
```
# utf8proc
# oatpp

# after intall oatpp, need set lib and include path (set actual path in your env):

# export LIBRARY_PATH=/usr/local/lib/oatpp-1.3.0:$LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/oatpp-1.3.0:$LD_LIBRARY_PATH
# export CPLUS_INCLUDE_PATH=/usr/local/include/oatpp-1.3.0/oatpp:$CPLUS_INCLUDE_PATH

```

### Build
To build the dynamic library for usage from e.g. Golang:
```sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make
cd ..
```

To build the native binaries, like the example server, with static libraries, run:
```sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
make
cd ..
```


### Run sample main
```sh
# ./build/bin/bert-main -m models/bert-base-uncased/ggml-model-f32.bin

# bertencoder_load_from_file: loading model from 'models/bert-base-uncased/ggml-model-f32.bin' - please wait ...
# bertencoder_load_from_file: n_vocab = 30522
# bertencoder_load_from_file: max_position_embeddings   = 512
# bertencoder_load_from_file: intermediate_size  = 3072
# bertencoder_load_from_file: num_attention_heads  = 12
# bertencoder_load_from_file: num_hidden_layers  = 12
# bertencoder_load_from_file: pad_token_id  = 0
# bertencoder_load_from_file: n_embd  = 768
# bertencoder_load_from_file: f16     = 0
# bertencoder_load_from_file: ggml ctx size = 417.73 MB
# bertencoder_load_from_file: ......................... done
# bertencoder_load_from_file: model size =   417.65 MB / num tensors = 201
# bertencoder_load_from_file: mem_per_token 0 KB, mem_per_input 0 MB
# main: number of tokens in prompt = 7


# main:    load time =   156.61 ms
# main:    eval time =    32.76 ms / 4.68 ms per token
# main:    total time =   189.38 ms

```

### Start rest server
```sh
./build/bin/bert-rest -m models/bert-base-uncased/ggml-model-f32.bin --port 8090

# bertencoder_load_from_file: loading model from 'models/bert-base-uncased/ggml-model-f32.bin' - please wait ...
# bertencoder_load_from_file: n_vocab = 30522
# bertencoder_load_from_file: max_position_embeddings   = 512
# bertencoder_load_from_file: intermediate_size  = 3072
# bertencoder_load_from_file: num_attention_heads  = 12
# bertencoder_load_from_file: num_hidden_layers  = 12
# bertencoder_load_from_file: pad_token_id  = 0
# bertencoder_load_from_file: n_embd  = 768
# bertencoder_load_from_file: f16     = 0
# bertencoder_load_from_file: ggml ctx size = 417.73 MB
# bertencoder_load_from_file: ......................... done
# bertencoder_load_from_file: model size =   417.65 MB / num tensors = 201
# bertencoder_load_from_file: mem_per_token 0 KB, mem_per_input 0 MB

#  I |2023-11-05 00:05:29 1699113929846361| MyApp:Server running on port 8090
```


### Converting models to ggml format
Converting models is similar to llama.cpp. Use models/bert-classify-to-ggml.py to make hf models into either f32 or f16 ggml models. 

```sh
cd models
# Clone a model from hf
git clone https://huggingface.co/sentence-transformers/bert-base-uncased
# Run conversions to 4 ggml formats (f32, f16)
sh run_conversions.sh bert-base-uncased 0
```

