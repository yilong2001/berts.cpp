#include "tokenization.h"
#include "bertbase.h"
#include "bertbasemodel.h"
#include "bertencoder.h"

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


//
// Loading and setup
//

struct BertBaseCtx * bertencoder_load_from_file(const char *fname) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname);

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname);
        return nullptr;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname);
            return nullptr;
        }
    }

    BertBaseCtx * bert_base_ctx = new BertBaseCtx;

    BertClassifierModel *model = new BertClassifierModel();

    bert_base_ctx->model = model;
    BertVocab & vocab = bert_base_ctx->vocab;

    // load hparams
    {
        auto &hparams = bert_base_ctx->hparams;

        fin.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *)&hparams.max_position_embeddings, sizeof(hparams.max_position_embeddings));
        fin.read((char *)&hparams.intermediate_size, sizeof(hparams.intermediate_size));
        fin.read((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        fin.read((char *)&hparams.num_hidden_layers, sizeof(hparams.num_hidden_layers));
        fin.read((char *)&hparams.pad_token_id, sizeof(hparams.pad_token_id));
        fin.read((char *)&hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char *)&hparams.n_labels, sizeof(hparams.n_labels));
        fin.read((char *)&hparams.num_beams, sizeof(hparams.num_beams));
        fin.read((char *)&hparams.f16, sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: max_position_embeddings   = %d\n", __func__, hparams.max_position_embeddings);
        printf("%s: intermediate_size  = %d\n", __func__, hparams.intermediate_size);
        printf("%s: num_attention_heads  = %d\n", __func__, hparams.num_attention_heads);
        printf("%s: num_hidden_layers  = %d\n", __func__, hparams.num_hidden_layers);
        printf("%s: pad_token_id  = %d\n", __func__, hparams.pad_token_id);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        std::unordered_map<std::string, uint64_t> *_vocab = new std::unordered_map<std::string, uint64_t>();
        int32_t n_vocab = bert_base_ctx->hparams.n_vocab;

        std::string word;
        for (int i = 0; i < n_vocab; i++)
        {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));

            word.resize(len);
            fin.read((char *)word.data(), len);

            vocab.token_to_id[word] = i;
            vocab._id_to_token[i] = word;

            (*_vocab)[word] = ((uint64_t)(i));
        }

        bert_base_ctx->tokenizer = new FullTokenizer(_vocab, false);
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (bert_base_ctx->hparams.f16) {
    case 0:
        wtype = GGML_TYPE_F32;
        break;
    case 1:
        wtype = GGML_TYPE_F16;
        break;
    case 2:
        wtype = GGML_TYPE_Q4_0;
        break;
    case 3:
        wtype = GGML_TYPE_Q4_1;
        break;
    default:
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname, bert_base_ctx->hparams.f16);
        bert_free(bert_base_ctx);
        return nullptr;
    }
    }

    size_t model_mem_req = 0;

    {
        const auto &hparams = bert_base_ctx->hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.num_hidden_layers;
        const int max_position_embeddings = hparams.max_position_embeddings;
        const int n_vocab = hparams.n_vocab;
        const int intermediate_size = hparams.intermediate_size;
        const int n_labels = hparams.n_labels;

        // Calculate size requirements

        model_mem_req += n_embd * n_vocab * ggml_type_sizef(wtype); // word_embeddings
        model_mem_req += n_embd * max_position_embeddings * ggml_type_sizef(wtype); // position_embeddings

        model_mem_req += 2 * n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_e_*

        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_*

        model_mem_req += 4 * n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // kqvo weights
        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // kqvo bias

        model_mem_req += 2 * n_layer * (n_embd * intermediate_size * ggml_type_sizef(wtype)); // ff_*_w
        model_mem_req += n_layer * (intermediate_size * ggml_type_sizef(GGML_TYPE_F32)); // ff_i_b
        model_mem_req += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ff_o_b

        model_mem_req += n_embd * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // classifier
        model_mem_req += n_labels * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // classifier

        model_mem_req += (5 + 16 * n_layer) * 512; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, model_mem_req / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = model_mem_req * 4,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        model->_ctx  = ggml_init(params);
        if (!model->_ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            bert_free(bert_base_ctx);
            return nullptr;
        }
    }

    auto &ctx = model->_ctx;
    // prepare memory for the weights
    {
        const auto &hparams = bert_base_ctx->hparams;

        const int n_embd = hparams.n_embd;
        const int max_position_embeddings = hparams.max_position_embeddings;
        const int n_vocab = hparams.n_vocab;

        const int n_layers = hparams.num_hidden_layers;
        const int intermediate_size = hparams.intermediate_size;
        const int n_labels = hparams.n_labels;
        const int n_token_type = hparams.n_token_type;

        model->bert.layers.resize(n_layers);

        model->bert.embeddings.word_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model->bert.embeddings.position_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, max_position_embeddings);
        model->bert.embeddings.token_type_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_token_type);

        model->bert.embeddings.ln_e_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model->bert.embeddings.ln_e_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        // map by name
        model->tensors["bert.embeddings.word_embeddings.weight"] = model->bert.embeddings.word_embeddings;
        model->tensors["bert.embeddings.position_embeddings.weight"] = model->bert.embeddings.position_embeddings;
        model->tensors["bert.embeddings.token_type_embeddings.weight"] = model->bert.embeddings.token_type_embeddings;

        model->tensors["bert.embeddings.LayerNorm.weight"] = model->bert.embeddings.ln_e_w;
        model->tensors["bert.embeddings.LayerNorm.bias"] = model->bert.embeddings.ln_e_b;

        for (int i = 0; i < n_layers; ++i)
        {
            auto &layer = model->bert.layers[i];

            layer.ln_att_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_att_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.q_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.k_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.v_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.o_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ff_i_w = ggml_new_tensor_2d(ctx, wtype, n_embd, intermediate_size);
            layer.ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, intermediate_size);

            layer.ff_o_w = ggml_new_tensor_2d(ctx, wtype, intermediate_size, n_embd);
            layer.ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.query.weight"] = layer.q_w;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.query.bias"] = layer.q_b;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.key.weight"] = layer.k_w;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.key.bias"] = layer.k_b;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.value.weight"] = layer.v_w;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.self.value.bias"] = layer.v_b;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.output.dense.weight"] = layer.o_w;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.output.dense.bias"] = layer.o_b;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight"] = layer.ln_att_w;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias"] = layer.ln_att_b;

            model->tensors["bert.encoder.layer." + std::to_string(i) + ".intermediate.dense.weight"] = layer.ff_i_w;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".intermediate.dense.bias"] = layer.ff_i_b;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".output.dense.weight"] = layer.ff_o_w;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".output.dense.bias"] = layer.ff_o_b;

            model->tensors["bert.encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight"] = layer.ln_out_w;
            model->tensors["bert.encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias"] = layer.ln_out_b;
        }

        model->pool_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
        model->pool_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model->cls_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_labels);
        model->cls_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_labels);

        model->tensors["bert.pooler.dense.weight"] = model->pool_w;
        model->tensors["bert.pooler.dense.bias"] = model->pool_b;
        model->tensors["classifier.weight"] = model->cls_w;
        model->tensors["classifier.bias"] = model->cls_b;
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            if (fin.eof())
            {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i)
            {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model->tensors.find(name.data()) == model->tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                bert_free(bert_base_ctx);
                return nullptr;
            }

            auto tensor = model->tensors[name.data()];
            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                bert_free(bert_base_ctx);
                return nullptr;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%lld, %lld]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                bert_free(bert_base_ctx);
                return nullptr;
            }

            if (0)
            {
                static const char *ftype_str[] = {
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                };
                printf("%24s - [%5lld, %5lld], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            size_t bpe = 0;

            switch (ftype)
            {
            case 0:
                bpe = ggml_type_size(GGML_TYPE_F32);
                break;
            case 1:
                bpe = ggml_type_size(GGML_TYPE_F16);
                break;
            case 2:
                bpe = ggml_type_size(GGML_TYPE_Q4_0);
                assert(ne[0] % 64 == 0);
                break;
            case 3:
                bpe = ggml_type_size(GGML_TYPE_Q4_1);
                assert(ne[0] % 64 == 0);
                break;
            default:
            {
                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                bert_free(bert_base_ctx);
                return nullptr;
            }
            };

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %llu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                bert_free(bert_base_ctx);
                return nullptr;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0)
            {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }

    fin.close();

    // Calculate space requirements for setting up context buffers later
    {
        const auto &hparams = bert_base_ctx->hparams;

        // bert_vocab_id tokens[] = {0, 1, 2, 3};
        // TODO: We set the initial buffer size to 32MB and hope it's enough. Maybe there is a better way to do this?
        bert_base_ctx->buf_compute.resize(hparams.MEM_SIZE);
        //bert_eval(bert_base_ctx, &hparams, 1, tokens, 4, nullptr);
        bert_base_ctx->max_batch_n = hparams.MAX_BATCH_N;

        // TODO: Max tokens should be a param?
        int32_t N = hparams.max_position_embeddings;
        bert_base_ctx->mem_per_input = 1.1 * (bert_base_ctx->mem_per_token * N); // add 10% to account for ggml object overhead
    }
    printf("%s: mem_per_token %zu KB, mem_per_input %lld MB\n", __func__, bert_base_ctx->mem_per_token / (1 << 10), bert_base_ctx->mem_per_input / (1 << 20));

    return bert_base_ctx;
}

} // namespace end

