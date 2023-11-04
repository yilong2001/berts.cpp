#ifndef BERTBASE_H
#define BERTBASE_H

#include "tokenization.h"

#include <cstring>
#include <cmath>
#include <ggml.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include <map>
#include <string>

namespace bert {


#define bert_vocab_id int32_t

void print_ggml_tensor(const char *info, ggml_tensor *tensor);

#define BERT_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "BERT_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

class BertHiParams {
public:
    BertHiParams() {}

    int32_t n_vocab = 119547;
    int32_t max_position_embeddings = 512;
    int32_t hidden_dim = 3072;
    int32_t n_heads = 12;
    int32_t n_layers = 6;
    int32_t pad_token_id = 0;
    int32_t bos_token_id = 0;
    int32_t eos_token_id = 2;
    int32_t n_embd = 768;
    int32_t n_labels = 2;
    int32_t f16 = 1;
    int32_t intermediate_size = 3072;
    int32_t num_attention_heads = 12;
    int32_t num_hidden_layers = 12;
    int32_t num_beams = 1;
    float layer_norm_eps = 1e-12;

    int32_t n_token_type = 2;
    
    float  f_norm_eps = 1e-12;

    int32_t CLS = 101;
    int32_t MASK = 103;
    int32_t PAD = 0;
    int32_t SEP = 102;
    int32_t UNK = 100;

    static constexpr size_t MEM_SIZE = 512 * 1024 * 1024;
    static constexpr size_t MAX_BATCH_N = 4;
};

class BertParams {
public:
    BertParams(){}
    
    int32_t n_threads = 6;
    int32_t port = 8080; // server mode port to bind

    const char* model = ""; // model path
    const char* prompt = "";
    const char* vocab = ""; // vocab file
};


class BertVocab {
public:
    BertVocab() {}

    std::map<std::string, bert_vocab_id> token_to_id;

    std::map<bert_vocab_id, std::string> _id_to_token;
};


class BertBuffer {
public:
    BertBuffer() { data = NULL; size = 0; }

    uint8_t * data = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~BertBuffer() {
        delete[] data;
    }
};

class AbstractBertModel {
public:
    virtual ggml_tensor* forward(BertHiParams *hparams, ggml_context *ctx0,  bert_vocab_id *tokens, int N) = 0;
};


class BertBaseCtx {
public:
    BertBaseCtx() {}

    BertHiParams hparams;

    FullTokenizer *tokenizer = NULL;
    AbstractBertModel *model = NULL;
    BertVocab vocab;

    size_t mem_per_token;
    int64_t mem_per_input;
    int32_t max_batch_n;
    BertBuffer buf_compute;

    ~BertBaseCtx() {
        if (tokenizer != NULL) {
            delete tokenizer;
            tokenizer = NULL;          
        }
        if (model != NULL) {
            delete model;
            model = NULL;          
        }
    }
};

void bert_free(BertBaseCtx * ctx);

bool bert_params_parse(int argc, char **argv, BertParams &params);

void bert_encode_classify(
    struct BertBaseCtx * ctx,
    int32_t n_threads,
    const char * texts,
    float * labels);

// n_batch_size - how many to process at a time
// n_inputs     - total size of texts and embeddings arrays
void bert_encode_batch_classify(
    struct BertBaseCtx * ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    int32_t n_inputs,
    const char ** texts,
    float ** labels);

// Api for separate tokenization & eval

void bert_eval_classify(
    struct BertBaseCtx * ctx,
    int32_t n_threads,
    bert_vocab_id *tokens,
    int32_t n_tokens,
    float * labels);

// NOTE: for batch processing the longest input must be first
void bert_eval_batch_classify(
    struct BertBaseCtx * ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    bert_vocab_id ** batch_tokens,
    int32_t * n_tokens,
    float ** labels);

int32_t bert_n_embd(BertBaseCtx * ctx);
int32_t bert_n_max_tokens(BertBaseCtx * ctx);

const char* bert_vocab_id_to_token(BertBaseCtx * ctx, bert_vocab_id id);
const size_t bert_tokens_to_ids(BertBaseCtx * ctx, std::vector<std::string> token_chars, size_t max_length, bert_vocab_id *outids);

}

#endif // DISTILBERT_H
