#ifndef BERTENCODER_H
#define BERTENCODER_H

#include "bertbase.h"
#include "bertbasemodel.h"

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


class BertClassifierModel : public AbstractBertModel {
public:
    // embeddings weights
    // transformer attentions
    BertEncoderBert bert;

    ggml_tensor *pool_w;
    ggml_tensor *pool_b;

    ggml_tensor *cls_w;
    ggml_tensor *cls_b;

    ggml_context *_ctx = NULL;
    std::map<std::string, ggml_tensor *> tensors;

    void set_ggml_context(ggml_context *ctx_) { _ctx = ctx_; }
    ggml_context* get_ggml_context() { return _ctx; }

    ggml_tensor* forward(BertHiParams *hparams, ggml_context *ctx0,  bert_vocab_id *tokens, int N) {
        BERT_ASSERT(cls_w != NULL);
        BERT_ASSERT(cls_b != NULL);

        struct ggml_tensor *inpL = bert.forward(hparams, ctx0, tokens, N);

        struct ggml_tensor *cur = ggml_view_2d(ctx0, inpL, inpL->ne[0], inpL->ne[2], inpL->ne[1]*inpL->ne[0], 0);
        print_ggml_tensor("outitem ggml_view_2d 1 ", cur);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, pool_w, cur), pool_b);
        print_ggml_tensor("pool liner ", cur);

        cur = ggml_tanh(ctx0, cur);
        print_ggml_tensor("pool tanh ", cur);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, cls_w, cur), cls_b);
        print_ggml_tensor("classifier liner ", cur);

        cur = ggml_soft_max(ctx0, cur);
        print_ggml_tensor("classifier softmax ", cur);

        return cur;
    }

    ~BertClassifierModel() {
        if (_ctx != NULL) {
            ggml_free(_ctx);
            _ctx = NULL;
        }
    }
};



struct BertBaseCtx * bertencoder_load_from_file(const char * fname);


}

#endif // BERTENCODER_H
