#include "models.h"

ggml_cgraph * clip_graph_whisper_enc::build() {
    const int n_frames = img.nx();

    // Qwen2_5OmniAudioEncoder with n_window from the mmproj (jina-embeddings-v5-omni): the variable-length
    // mel is cut into n_window*2-frame chunks, each chunk gets its own conv stack (zero-padded at both of
    // its ends) and its own positions 0..n_window-1, and attention stays inside the chunk. A shorter last
    // chunk zero-padded by the convs gives the same valid outputs as torch's padded_mask path.
    const int  n_window = proj_type == PROJECTOR_TYPE_QWEN2A ? hparams.audio_n_window : 0;
    const bool chunked  = n_window > 0;
    const int  n_pos    = chunked ? (n_frames - 1) / 2 + 1 : n_frames / 2;
    GGML_ASSERT(model.position_embeddings->ne[1] >= (chunked ? n_window : n_pos));

    ggml_tensor * inp = build_inp_raw(1);

    if (chunked) {
        const int chunk_len = n_window * 2;
        ggml_tensor * out = nullptr;
        for (int off = 0; off < n_frames; off += chunk_len) {
            const int len = std::min(chunk_len, n_frames - off);
            ggml_tensor * cur = ggml_cont(ctx0, ggml_view_3d(ctx0, inp, len, inp->ne[1], inp->ne[2],
                                                             inp->nb[1], inp->nb[2], (size_t) off * inp->nb[0]));
            cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, cur, 1, 1);
            cur = ggml_gelu_erf(ctx0, ggml_add(ctx0, cur, model.conv1d_1_b));
            cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
            cur = ggml_gelu_erf(ctx0, ggml_add(ctx0, cur, model.conv1d_2_b));
            out = out ? ggml_concat(ctx0, out, cur, 0) : cur;
        }
        GGML_ASSERT(out->ne[0] == n_pos);
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, out));
        cb(inp, "after_conv1d", -1);
    } else {
        // convolution + gelu
        ggml_tensor * cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_1_b);

        cur = ggml_gelu_erf(ctx0, cur);

        cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_2_b);

        cur = ggml_gelu_erf(ctx0, cur);
        // transpose
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cb(inp, "after_conv1d", -1);
    }

    // sanity check (only check one layer, but it should be the same for all)
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);
    GGML_ASSERT(!model.layers[0].k_b); // no bias for k

    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], chunked ? n_window : n_pos,
        model.position_embeddings->nb[1], 0
    );
    build_vit_opts vit_opts;
    if (chunked) {
        // positions restart at 0 in every chunk
        const int n_chunks = (n_pos + n_window - 1) / n_window;
        pos_embd_selected = ggml_repeat_4d(ctx0, pos_embd_selected, pos_embd_selected->ne[0], (int64_t) n_chunks * n_window, 1, 1);
        pos_embd_selected = ggml_view_2d(ctx0, pos_embd_selected, pos_embd_selected->ne[0], n_pos, pos_embd_selected->nb[1], 0);

        ggml_tensor * chunk_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
        ggml_set_name(chunk_mask, "audio_chunk_mask");
        ggml_set_input(chunk_mask);
        vit_opts.attn_mask = chunk_mask;
    }
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            pos_embd_selected,
                            nullptr,
                            vit_opts);

    cb(cur, "after_transformer", -1);

    if (model.audio_has_stack_frames()) {
        // StackAudioFrames
        // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
        cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
        cb(cur, "after_stacked", -1);
    }

    if (proj_type == PROJECTOR_TYPE_ULTRAVOX) {
        // UltravoxProjector
        // pre-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);

        // ffn in
        cur = build_mm(model.mm_1_w, cur);

        // swiglu
        // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
        cur = ggml_swiglu_swapped(ctx0, cur);

        // mid-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_mid_w);

        // ffn out
        cur = build_mm(model.mm_2_w, cur);

    } else if (proj_type == PROJECTOR_TYPE_QWEN2A) {
        // projector
        cur = build_mm(model.mm_fc_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_fc_b);

    } else if (proj_type == PROJECTOR_TYPE_VOXTRAL) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MUSIC_FLAMINGO) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MERALION) {
        // stack (above) -> ln -> linear0+silu -> GLU -> out
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
        cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);

        cur = ggml_mul_mat(ctx0, model.mm_0_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_0_b);
        cur = ggml_silu(ctx0, cur);

        ggml_tensor * gate = ggml_mul_mat(ctx0, model.mm_1_w, cur);
        gate = ggml_add(ctx0, gate, model.mm_1_b);
        gate = ggml_silu(ctx0, gate);

        ggml_tensor * pool = ggml_mul_mat(ctx0, model.mm_2_w, cur);
        pool = ggml_add(ctx0, pool, model.mm_2_b);

        cur = ggml_mul(ctx0, gate, pool);

        cur = ggml_mul_mat(ctx0, model.mm_3_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_3_b);

    } else if (proj_type == PROJECTOR_TYPE_GLMA) {
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
            cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);
            cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
            cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_2_w, model.mm_2_b, hparams.ffn_op, 0);
            cur = ggml_concat(ctx0, model.mm_boi, cur, 1);
            cur = ggml_concat(ctx0, cur, model.mm_eoi, 1);
    } else {
        GGML_ABORT("%s: unknown projector type", __func__);
    }

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
