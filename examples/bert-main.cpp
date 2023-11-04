#include "tokenization.h"
#include "bertbase.h"
#include "bertencoder.h"
#include <ggml.h>

#include <unistd.h>
#include <stdio.h>
#include <vector>

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bert::BertParams params;

    if (bert::bert_params_parse(argc, argv, params) == false) {
        return 1;
    }

    int64_t t_load_us = 0;

    bert::BertBaseCtx *ctx;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if ((ctx = bert::bertencoder_load_from_file(params.model)) == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int64_t t_eval_us  = 0;
    int64_t t_start_us = ggml_time_us();
    int N = bert::bert_n_max_tokens(ctx);
    // tokenize the prompt
    std::vector<std::string> token_chars;
    ctx->tokenizer->tokenize("分类问题...", &token_chars, N-2);
    printf("%s: number of tokens in prompt = %zu\n", __func__, token_chars.size());

    int batch_num = 1;

    std::vector<bert_vocab_id> buf_tokens;
    buf_tokens.resize(batch_num * N);

    std::vector<bert_vocab_id*> org_tokens(batch_num);
    bert_vocab_id* it_tokens = buf_tokens.data();
    org_tokens[0] = it_tokens;
    //org_tokens[1] = (it_tokens+N);

    std::vector<int> n_tokens;
    n_tokens.resize(batch_num);

    n_tokens[0] = bert::bert_tokens_to_ids(ctx, token_chars, N, it_tokens);
    //n_tokens[1] = distilbert_tokens_to_ids(ctx, &(ctx->model.hparams), token_chars, N, (it_tokens+N));

    //printf("\ndecode : \n");
    //for (auto& tok : buf_tokens) {
    //    printf("%d -> %s\n", tok, bert::bert_vocab_id_to_token(ctx, tok));
    //}

    std::vector<float> labels(1*ctx->hparams.n_labels);
    std::vector<float*> org_labels(1);
    org_labels[0] = labels.data();
    //org_labels[1] = labels.data()+ctx->model.hparams.n_labels;

    //bert_eval_classify(ctx, &(ctx->model.hparams), params.n_threads, it_tokens, n_tokens[0], labels.data());
    bert::bert_eval_batch_classify(ctx, params.n_threads, 1, org_tokens.data(), n_tokens.data(), org_labels.data());
    t_eval_us += ggml_time_us() - t_start_us;
    
    printf("[");
    for(auto e : labels) {
        printf("%1.4f, ", e);
    }
    printf("]\n");

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        //printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:  eval time = %8.2f ms / %.2f ms per token\n", __func__, t_eval_us/1000.0f, t_eval_us/1000.0f/token_chars.size());
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    bert::bert_free(ctx);

    return 0;
}
