#include "tokenization.h"
#include "bertbase.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <limits>

namespace bert {

static std::string shape_to_string(ggml_tensor *tensor) {
    std::ostringstream oss;
    oss << '[';
    for (int i = tensor->n_dims - 1; i >= 0; i--) {
        oss << tensor->ne[i] << (i > 0 ? ", " : "");
    }
    oss << ']';
    return oss.str();
}

static std::string strides_to_string(ggml_tensor *tensor) {
    std::ostringstream oss;
    oss << '[';
    for (int i = tensor->n_dims - 1; i >= 0; i--) {
        oss << tensor->nb[i] << (i > 0 ? ", " : "");
    }
    oss << ']';
    return oss.str();
}

std::string to_string(ggml_tensor *tensor, bool with_data) {
    std::ostringstream oss;
    oss << "ggml_tensor(";

    if (with_data) {
        if (tensor->n_dims > 3)
            oss << "[";
        for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
            if (tensor->n_dims > 2)
                oss << (i3 > 0 ? ",\n\n[" : "[");
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                if (tensor->n_dims > 1)
                    oss << (i2 > 0 ? ",\n\n[" : "[");
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    oss << (i1 > 0 ? ",\n[" : "[");
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        auto ptr = (char *)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                                   i0 * tensor->nb[0];
                        //float val;
                        oss << ", ";
                        if (tensor->type == GGML_TYPE_F32) {
                            oss <<  *(float *)ptr;
                        } else if (tensor->type == GGML_TYPE_F16) {
                            oss <<  ggml_fp16_to_fp32(*(ggml_fp16_t *)ptr);
                        } else if (tensor->type == GGML_TYPE_I32) {
                            oss <<  *(int *)ptr;
                        } else {
                            oss << "unimplemented";
                        }
                        //oss << (i0 > 0 ? ", " : "") << std::setw(7) << std::fixed << std::setprecision(4) << val;
                    }
                    oss << "]";
                }
                if (tensor->n_dims > 1)
                    oss << "]";
            }
            if (tensor->n_dims > 2)
                oss << "]";
        }
        if (tensor->n_dims > 3)
            oss << "]";
        oss << ", ";
    }

    oss << "shape=" << shape_to_string(tensor) << ", stride=" << strides_to_string(tensor) << ")";
    return oss.str();
}

void print_ggml_tensor(const char *info, ggml_tensor *tensor) {
#ifdef _DEBUG_
    std::cout<<std::endl<<info<<" : "<<std::endl;
    printf ("ne[0]:%d, ne[1]:%d, ne[2]:%d, ne[3]:%d\n", tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    printf ("nb[0]:%d, nb[1]:%d, nb[2]:%d, nb[3]:%d\n", tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
#endif
}

void bert_print_usage(char **argv, const BertParams &params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  --vocab FNAME         vocab file path \n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  --port p     port to bind in server mode (default: %d)\n", params.port);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model);
    fprintf(stderr, "\n");
}

int32_t bert_n_embd(BertBaseCtx * ctx) {
    return ctx->hparams.n_embd;
}

int32_t bert_n_max_tokens(BertBaseCtx * ctx) {
    return ctx->hparams.max_position_embeddings;
}

bool bert_params_parse(int argc, char **argv, BertParams &params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "--vocab") {
            params.vocab = argv[++i];
        } else if (arg == "--port") {
            params.port = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            bert_print_usage(argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bert_print_usage(argv, params);
            exit(0);
        }
    }

    return true;
}


void bert_resize_ctx(BertBaseCtx * ctx, int32_t new_size) {    
    int64_t buf_size_new = ctx->mem_per_input * new_size;

    // TODO: Max memory should be a param? Now just 1 GB
    int64_t GB = 1 << 30;
    //printf("%s: requested_buf_size %lldMB\n", __func__, buf_size_new / (1 << 20));
    if (buf_size_new > GB) {
        int32_t adjusted_new_size = GB / ctx->mem_per_input;
        if (adjusted_new_size < 1) adjusted_new_size = 1;
        //printf("%s: requested batch size %d, actual new batch size %d\n", __func__, new_size, adjusted_new_size);
        new_size = adjusted_new_size;
        buf_size_new = ctx->mem_per_input * new_size;
    }

    if (new_size > ctx->max_batch_n) {
        ctx->buf_compute.resize(buf_size_new);
        ctx->max_batch_n = new_size;
    }
}

void bert_free(BertBaseCtx * ctx) {
    delete ctx;
}

// Main api, does both tokenizing and evaluation

const char* bert_vocab_id_to_token(BertBaseCtx * ctx, bert_vocab_id id) {
    BertVocab & vocab = ctx->vocab;
    auto it = vocab._id_to_token.find(id);
    if (it != vocab._id_to_token.end())
    {
        return it->second.c_str();
    }
    return "[UNK]";
}

const size_t bert_tokens_to_ids(BertBaseCtx * ctx, std::vector<std::string> token_chars, size_t max_length, bert_vocab_id *outids) {
    const BertHiParams *hparams = &ctx->hparams;
    auto *token_map = &ctx->vocab.token_to_id;
    size_t index = 0;
    outids[index++] = hparams->CLS;
    //for(auto tokenit = token_chars.begin(); tokenit != token_chars.end(); tokenit++) {
    for (int i = 0; i < token_chars.size(); ++i) {
        auto tokenit = token_chars[i];
        if (index >= max_length - 1) {
            return index;
        }
        auto it = token_map->find(tokenit);
        if (it != token_map->end()) {
           outids[index++] = it->second;
        } else {
            if ((tokenit).compare("[UNK]")) {
                outids[index++] = hparams->UNK;
            } else if ((tokenit).compare("[PAD]")) {
                outids[index++] = hparams->PAD;
            } else if ((tokenit).compare("[MASK]")) {
                outids[index++] = hparams->MASK;
            } else {
                outids[index++] = hparams->UNK;
            }
        }
    }

    outids[index++] = hparams->SEP;
    //for ( ; index < max_length; ) {
    //    outids[index++] = hparams->PAD;
    //}
    
    return index;
}

void bert_encode_classify(
    struct BertBaseCtx *ctx,
    int32_t n_threads,
    const char *texts,
    float *labels)
{
    bert_encode_batch_classify(ctx, n_threads, 1, 1, &texts, &labels);
}

void bert_encode_batch_classify(
    struct BertBaseCtx *ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    int32_t n_inputs,
    const char ** texts,
    float **labels)
{
    const BertHiParams *hparams = &ctx->hparams;
    // TODO: Disable batching for now
    if (n_batch_size > ctx->max_batch_n) {
        n_batch_size = ctx->max_batch_n;
    }

    int32_t N = bert_n_max_tokens(ctx);
    auto &tokenizer = ctx->tokenizer;

    std::vector<bert_vocab_id> buf_tokens;
    // Most of this buffer will be unused in typical case where inputs are not that long.
    buf_tokens.resize(N * n_inputs);
    std::vector<int32_t> n_tokens = std::vector<int32_t>(n_inputs);
    std::vector<bert_vocab_id*> unsorted_tokens(n_inputs);
    bert_vocab_id* it_tokens = buf_tokens.data();
    for (int i = 0; i < n_inputs; i++) {
        std::vector<std::string> token_chars;
        ctx->tokenizer->tokenize(texts[i], &token_chars, N-2);

        unsorted_tokens[i] = it_tokens;
        
        n_tokens[i] = bert_tokens_to_ids(ctx, token_chars, N, it_tokens);
        it_tokens += n_tokens[i];
    }

    if (n_batch_size == n_inputs) {
        bert_eval_batch_classify(ctx, n_threads, n_batch_size, unsorted_tokens.data(), n_tokens.data(), labels);
    } else if (n_batch_size > n_inputs) {
        bert_eval_batch_classify(ctx, n_threads, n_inputs, unsorted_tokens.data(), n_tokens.data(), labels);
    } else {
        for (int i = 0; i < n_inputs; i += n_batch_size) {
            if (i + n_batch_size > n_inputs) {
                n_batch_size = n_inputs - i;
            }
            bert_eval_batch_classify(ctx, n_threads, n_batch_size, &unsorted_tokens[i], &n_tokens[i], &labels[i]);
        }
    }
}



void bert_eval_classify(
    struct BertBaseCtx *ctx,
    int32_t n_threads,
    bert_vocab_id *tokens,
    int32_t n_tokens,
    float *labels)
{
    bert_eval_batch_classify(ctx, n_threads, 1, &tokens, &n_tokens, labels ? &labels : nullptr);
}

void bert_eval_batch_classify(
    BertBaseCtx * ctx,
    int32_t n_threads,
    int32_t n_batch_size,
    bert_vocab_id **batch_tokens,
    int32_t * n_tokens,
    float ** batch_labels)
{
    AbstractBertModel *model = ctx->model;
    BertHiParams *hparams = &ctx->hparams;
    bool mem_req_mode = !batch_labels;
    // batch_embeddings is nullptr for the initial memory requirements run
    if (!mem_req_mode && n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size);
        if (n_batch_size > ctx->max_batch_n) {
            fprintf(stderr, "%s: tried to increase buffers to batch size %d but failed\n", __func__, n_batch_size);
            return;
        }
    }

    // TODO: implement real batching: 
    for (int ba = 0; ba < n_batch_size; ba++)
    {
        const int N = n_tokens[ba];
        const auto &tokens = batch_tokens[ba];

        const int n_embd = hparams->n_embd;
        const int n_layer = hparams->n_layers;
        const int N_MAX = hparams->max_position_embeddings;
        const int n_head = hparams->n_heads;
        const int n_labels = hparams->n_labels;

        const int d_head = n_embd / n_head;
        const float norm_eps   = hparams->f_norm_eps;

        std::vector<float> result;
        if (N > N_MAX)
        {
            fprintf(stderr, "Too many tokens, maximum is %d\n", N_MAX);
            return;
        }

        auto & mem_per_token = ctx->mem_per_token;
        auto & buf_compute   = ctx->buf_compute;

        struct ggml_init_params params = {
            .mem_size = buf_compute.size,
            .mem_buffer = buf_compute.data,
            .no_alloc = false,
        };

        struct ggml_context *ctx0 = ggml_init(params);
        struct ggml_cgraph gf = {};

        // Embeddings. word_embeddings + position_embeddings
        // layers

        ggml_tensor *output = model->forward(hparams, ctx0, tokens, N);

        // run the computation
        ggml_build_forward_expand(&gf, output);
        ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

        std::cout<<to_string(output, true)<<std::endl;

        // float *dat = ggml_get_data_f32(output);
        // pretty_print_tensor(dat, output->ne, output->nb, output->n_dims - 1, "");

        #ifdef GGML_PERF
            // print timing information per ggml operation (for debugging purposes)
            // requires GGML_PERF to be defined
            ggml_graph_print(&gf);
        #endif

        if (!mem_req_mode) {
            memcpy(batch_labels[ba], (float *)ggml_get_data(output), sizeof(float) * n_labels);
        } else {
            mem_per_token = ggml_used_mem(ctx0) / N;

            // printf("used_mem = %zu KB \n", ggml_used_mem(ctx0) / 1024);
            printf("mem_per_token = %zu KB \n", mem_per_token / 1024);
        }

        ggml_free(ctx0);
    }
}



}
