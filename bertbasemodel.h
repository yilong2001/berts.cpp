#ifndef BERTBASEMODEL_H
#define BERTBASEMODEL_H

#include "tokenization.h"
#include "ggml.h"

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


class BertEmbedding {
public:
    BertEmbedding() {}
    ggml_tensor *word_embeddings = NULL;
    ggml_tensor *position_embeddings = NULL;
    ggml_tensor *token_type_embeddings = NULL;
    ggml_tensor *ln_e_w;
    ggml_tensor *ln_e_b;

    ggml_tensor* forward(BertHiParams *hparams, ggml_context *ctx0,  bert_vocab_id *tokens, int N) {
        BERT_ASSERT(word_embeddings != NULL);
        BERT_ASSERT(position_embeddings != NULL);
        BERT_ASSERT(ln_e_w != NULL);
        BERT_ASSERT(ln_e_b != NULL);

        float norm_eps   = hparams->f_norm_eps;

        struct ggml_tensor *token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        memcpy(token_layer->data, tokens, N * ggml_element_size(token_layer));
        //std::cout<<to_string(token_layer, true)<<std::endl;

        struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
        for (int i = 0; i < N; i++) {
            ggml_set_i32_1d(positions, i, i);
        }

        struct ggml_tensor *inpL = ggml_get_rows(ctx0, word_embeddings, token_layer);
        //printf("\n word_embeddings  \n");

        inpL = ggml_add(ctx0, ggml_get_rows(ctx0, position_embeddings, positions), inpL);
        //printf("\n position_embeddings  \n");

        if (token_type_embeddings != NULL) {
            struct ggml_tensor *token_type_0 = ggml_view_1d(ctx0, token_type_embeddings, token_type_embeddings->ne[0], 0);
            print_ggml_tensor("token_type_0 ", token_type_0);
            inpL = ggml_add(ctx0, inpL, token_type_0);
        }

        // embd norm
        {
            inpL = ggml_norm(ctx0, inpL, norm_eps);
            printf("\n embd norm begin : \n");

            inpL = ggml_mul(ctx0, inpL, ln_e_w);
            print_ggml_tensor("norm mul after inpL ", inpL);

            inpL = ggml_add(ctx0, inpL, ln_e_b);
            print_ggml_tensor("norm add after inpL ", inpL);
        }

        return inpL;
    }
};

class BertTransformer {
public:
    BertTransformer() {}

    // normalization
    ggml_tensor *ln_att_w;
    ggml_tensor *ln_att_b;

    ggml_tensor *q_w;
    ggml_tensor *q_b;
    ggml_tensor *k_w;
    ggml_tensor *k_b;
    ggml_tensor *v_w;
    ggml_tensor *v_b;

    ggml_tensor *o_w;
    ggml_tensor *o_b;

    // ffn
    ggml_tensor *ff_i_w;
    ggml_tensor *ff_i_b;

    ggml_tensor *ff_o_w;
    ggml_tensor *ff_o_b;

    ggml_tensor *ln_out_w;
    ggml_tensor *ln_out_b;

    ggml_tensor* forward(BertHiParams *hparams, ggml_context *ctx0,  ggml_tensor *hide_state, int N) {
        BERT_ASSERT(ln_att_w != NULL);
        BERT_ASSERT(ln_att_b != NULL);

        const int n_embd = hparams->n_embd;
        const int n_layer = hparams->n_layers;
        const int N_MAX = hparams->max_position_embeddings;
        const int n_head = hparams->n_heads;
        const int n_labels = hparams->n_labels;

        const int d_head = n_embd / n_head;
        const float norm_eps   = hparams->f_norm_eps;

        struct ggml_tensor *inpL = hide_state;
        struct ggml_tensor *cur = inpL;

        // a layer
        // self-attention
        {
            struct ggml_tensor *Qcur = cur;
            struct ggml_tensor *qLin = ggml_add(ctx0, ggml_mul_mat(ctx0, q_w, Qcur), q_b);
            print_ggml_tensor("self-attention qcur liner  ", qLin);

            Qcur = ggml_reshape_3d(ctx0, qLin,  d_head, n_head, N);
            print_ggml_tensor("self-attention qcur rehsape  ", Qcur);

            //q = soft_max(q / sqrt(head width))
            Qcur = ggml_scale(ctx0, Qcur, ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head)));
            print_ggml_tensor("self-attention ScaleQ  ", Qcur);

            struct ggml_tensor *Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            print_ggml_tensor("self-attention qcur permute  ", Q);

            struct ggml_tensor *Kcur = cur;
            struct ggml_tensor *klin = ggml_add(ctx0, ggml_mul_mat(ctx0, k_w, Kcur), k_b);
            print_ggml_tensor("self-attention Kcur liner  ", klin);

            Kcur = ggml_reshape_3d(ctx0, klin,  d_head, n_head, N);
            print_ggml_tensor("self-attention Kcur rehsape  ", Kcur);

            struct ggml_tensor *K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
            print_ggml_tensor("self-attention Kcur permute  ", K);

            struct ggml_tensor *Vcur = cur;
            struct ggml_tensor *vlin = ggml_add(ctx0, ggml_mul_mat(ctx0, v_w, Vcur), v_b);
            print_ggml_tensor("self-attention Vcur lin  ", vlin);

            Vcur = ggml_reshape_3d(ctx0, vlin, d_head, n_head, N);
            print_ggml_tensor("self-attention Vcur rehsape  ", Vcur);

            struct ggml_tensor *V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
            print_ggml_tensor("self-attention Vcur permute  ", V);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
            print_ggml_tensor("self-attention KQ  ", KQ);

            KQ = ggml_soft_max(ctx0, KQ);
            print_ggml_tensor("self-attention KQ soft_max  ", KQ);

            V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
            print_ggml_tensor("self-attention transpose V  ", V);


            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ);
            print_ggml_tensor("self-attention KQV  ", KQV);

            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            print_ggml_tensor("self-attention KQV permute ", KQV);

            //cur = ggml_cpy(ctx0, KQV, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
            //print_ggml_tensor("self-attention KQV permute after cpy ", cur);

            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV), n_embd, N);
            print_ggml_tensor("self-attention KQV permute after cpy ", cur);
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, o_w, cur), o_b);
        print_ggml_tensor("attention output ", cur);

        // re-add the layer input
        cur = ggml_add(ctx0, cur, inpL);
        print_ggml_tensor("attention output add org ", cur);

        // attention norm
        {
            cur = ggml_norm(ctx0, cur, norm_eps);
            print_ggml_tensor("sa_layer_norm norm ", cur);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, ln_att_w), ln_att_b);
            print_ggml_tensor("sa_layer_norm norm liner ", cur);
        }

        struct ggml_tensor *att_output = cur;
        // intermediate_output = self.intermediate(attention_output)
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, ff_i_w, cur), ff_i_b);
        print_ggml_tensor("attention ffn in liner ", cur);

        cur = ggml_gelu(ctx0, cur);
        print_ggml_tensor("attention ffn gelu ", cur);

        // layer_output = self.output(intermediate_output, attention_output)
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, ff_o_w, cur), ff_o_b);
        print_ggml_tensor("attention ffn out liner ", cur);

        // attentions bypass the intermediate layer
        cur = ggml_add(ctx0, att_output, cur);
        print_ggml_tensor("attention ffn+before cur ", cur);

        // output norm
        {
            cur = ggml_norm(ctx0, cur, norm_eps);
            print_ggml_tensor("output_layer_norm norm ", cur);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, ln_out_w), ln_out_b);
            print_ggml_tensor("output_layer_norm norm liner ", cur);
        }

        return cur;
    }
};

class BertEncoderBert {
public:
    BertEncoderBert() {}

    BertEmbedding embeddings;
    std::vector<BertTransformer> layers;

    ggml_tensor* forward(BertHiParams *hparams, ggml_context *ctx0,  bert_vocab_id *tokens, int N) {
        // Embeddings. word_embeddings + position_embeddings + token type embedding
        struct ggml_tensor *inpL = embeddings.forward(hparams, ctx0, tokens, N);
        // layers
        for (int il = 0; il < layers.size(); il++) {
            inpL = layers[il].forward(hparams, ctx0, inpL, N);
        }

        return inpL;
    }
};


}

#endif // BERTBASEMODEL_H
