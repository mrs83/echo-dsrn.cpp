#include "models.h"

#include "llama-memory-recurrent.h"

//
// Echo-DSRN-Hybrid: Qwen2 backbone + DSRN memory injectors.
//
// Each injector is a purely additive residual block placed after transformer
// layer (j+1)*stride - 1.  It maintains two recurrent states (both F32, per
// sequence, carried in the hybrid memory's recurrent half):
//
//   h_t (fast state, dim D = n_embd)   — gated linear recurrence
//   c_t (slow state, dim D_s = dsrn_state_dim) — surprise-gated recurrence
//
// Reference: echo_hybrid/dsrn_memory_block.py (DSRNMemoryInjector).
//

void llama_model_echo_dsrn_hybrid::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen2::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DSRN_STATE_DIM,        hparams.dsrn_state_dim);
    ml.get_key(LLM_KV_DSRN_INJECTION_STRIDE, hparams.dsrn_injection_stride);

    if (hparams.dsrn_state_dim == 0) {
        throw std::runtime_error("echo-dsrn-hybrid: missing dsrn.state_dim");
    }
    if (hparams.dsrn_injection_stride == 0) {
        throw std::runtime_error("echo-dsrn-hybrid: missing dsrn.injection_stride");
    }
}

void llama_model_echo_dsrn_hybrid::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen2::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t D  = n_embd;
    const int64_t Ds = hparams.dsrn_state_dim;

    for (int i = 0; i < n_layer; ++i) {
        if ((i + 1) % (int) hparams.dsrn_injection_stride != 0) {
            continue;
        }

        auto & layer = layers[i];

        layer.dsrn_norm   = create_tensor(tn(LLM_TENSOR_DSRN_NORM, "weight", i), {D},       0);
        layer.dsrn_gru    = create_tensor(tn(LLM_TENSOR_DSRN_GRU,  "weight", i), {D, 3*D},  0);
        layer.dsrn_gru_b  = create_tensor(tn(LLM_TENSOR_DSRN_GRU,  "bias",   i), {3*D},     0);
        layer.dsrn_pred   = create_tensor(tn(LLM_TENSOR_DSRN_PRED, "weight", i), {D, D},    0);
        layer.dsrn_gate   = create_tensor(tn(LLM_TENSOR_DSRN_GATE, "weight", i), {D, Ds},   0);
        layer.dsrn_gate_b = create_tensor(tn(LLM_TENSOR_DSRN_GATE, "bias",   i), {Ds},      0);
        layer.dsrn_mem    = create_tensor(tn(LLM_TENSOR_DSRN_MEM,  "weight", i), {D, Ds},   0);
        layer.dsrn_mem_b  = create_tensor(tn(LLM_TENSOR_DSRN_MEM,  "bias",   i), {Ds},      0);
        layer.dsrn_lambda = create_tensor(tn(LLM_TENSOR_DSRN_LAMBDA, i), {Ds}, 0);
        layer.dsrn_read   = create_tensor(tn(LLM_TENSOR_DSRN_READ, "weight", i), {Ds, D},   0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_echo_dsrn_hybrid::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_echo_dsrn_hybrid::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // hybrid memory: attention KV cache (all layers) + recurrent state (injector layers)
    auto * inp = build_inp_mem_hybrid();
    auto * inp_attn = inp->get_attn();
    auto * inp_rs   = inp->get_recr();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t stride = hparams.dsrn_injection_stride;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_injector = ((il + 1) % stride) == 0;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        // NOTE: the DSRN injector after this layer needs the full sequence
        // (its recurrence advances once per token), so defer the last-layer
        // output-token crop until after the injector when this is one.
        if (il == n_layer - 1 && inp_out_ids && !is_injector) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        // DSRN memory injector (purely additive)
        if (is_injector) {
            cur = build_dsrn_injector(model, inp_rs, cur, il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            }
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// The DSRN injector.  Runs on the post-FFN residual stream of layer `il`.
//
// Fast state (h):  h_t = (1 - z_t) * h_{t-1} + z_t * r_t
//   z_t = sigmoid(gru[0:D]), r_t = tanh(gru[2D:3D]), gru = W_ih @ RMSNorm(x)
//
// Slow state (c):  c_t = (1 - g_t) * c_{t-1} + g_t * m_t
//   error    = mean_d(clamp((x - x_pred)^2, 10));  x_pred = W_pred @ h_{t-1}
//   surprise = error * softplus(lambda)
//   g_t      = sigmoid(W_gate @ h_t + b_gate + surprise)
//   m_t      = tanh(W_mem @ h_t + b_mem)
//
// Both scans map onto ggml_ssm_scan (a diagonal gated recurrence):
//
//   s_t = exp(softplus(dt_t) * A) * s_{t-1} + B_t * x_t * softplus(dt_t)
//
// with A = -1 and dt = the raw gate logits: decay = exp(-softplus(GL)) =
// sigmoid(-GL) = 1 - sigmoid(GL), and B = sigmoid(GL)/softplus(GL) with
// x = the candidate restores the exact product term.
//
// The recurrence is computed in fp32 (the reference upcasts the gate path to
// avoid bf16 saturation), and the result is cast back to the stream dtype.
ggml_tensor * llama_model_echo_dsrn_hybrid::graph::build_dsrn_injector(
        const llama_model & model,
        llm_graph_input_rs * inp,
        ggml_tensor * cur,
        int il) const {
    const auto * mctx_cur = inp->mctx;

    const auto & layer = model.layers[il];

    const int64_t D  = hparams.n_embd;
    const int64_t Ds = hparams.dsrn_state_dim;
    const int64_t T  = ubatch.n_seq_tokens;
    const int64_t S  = ubatch.n_seqs;

    const int64_t kv_head = mctx_cur->get_head();

    const ggml_type stream_type = cur->type;

    GGML_ASSERT(Ds == hparams.dsrn_state_dim);
    GGML_ASSERT(cur->ne[0] == D);

    // {D, n_tokens} => {D, T, S}; the injector computes in fp32
    cur = ggml_reshape_3d(ctx0, cur, D, T, S);
    ggml_tensor * x32 = ggml_cpy(ctx0, cur, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, T, S));
    cb(x32, "dsrn_in", il);

    // fast-state RMSNorm
    ggml_tensor * x_norm = build_norm(x32, layer.dsrn_norm, NULL, LLM_NORM_RMS, il);
    cb(x_norm, "dsrn_norm", il);

    // GRU projection: {3D, D} @ {D, T, S} => {3D, T, S}
    ggml_tensor * gru = ggml_mul_mat(ctx0, layer.dsrn_gru, x_norm);
    gru = ggml_add(ctx0, gru, layer.dsrn_gru_b);
    cb(gru, "dsrn_gru", il);

    // only rows [0:D) (z) and [2D:3D) (r) are used; rows [D:2D) are dead
    // weight retained for checkpoint compatibility
    ggml_tensor * z_logits = ggml_view_3d(ctx0, gru, D, T, S, gru->nb[1], gru->nb[2], 0);
    ggml_tensor * r_logits = ggml_view_3d(ctx0, gru, D, T, S, gru->nb[1], gru->nb[2], 2*D*ggml_element_size(gru));

    ggml_tensor * z = ggml_sigmoid(ctx0, z_logits);
    ggml_tensor * r = ggml_tanh(ctx0, r_logits);
    cb(z, "dsrn_z", il);
    cb(r, "dsrn_r", il);

    // ---- fast-state scan: h_t = (1 - z_t) * h_{t-1} + z_t * r_t ----
    // dt must be contiguous (ggml_ssm_scan requirement); the GRU row slice is a view
    ggml_tensor * dt      = ggml_cont(ctx0, z_logits);
    ggml_tensor * sp      = ggml_softplus(ctx0, z_logits);
    ggml_tensor * B       = ggml_div(ctx0, z, sp);
    ggml_tensor * x_scan  = ggml_reshape_4d(ctx0, r, 1, D, T, S);
    ggml_tensor * B_scan  = ggml_reshape_4d(ctx0, B, 1, D, T, S);

    ggml_tensor * A = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, D), -1.0f);
    ggml_tensor * C = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, D, T, S), 1.0f);

    ggml_tensor * h_states_all = mctx_cur->get_r_l(il);

    auto get_h_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
        ggml_tensor * ssm = ggml_reshape_4d(ctx, states, 1, 1, D, mctx_cur->get_size());
        return ggml_ssm_scan(ctx, ssm, x_scan, dt, A, B_scan, C, ids, /*K=*/1);
    };

    // h_scan: [y (D*T*S) | final state (D*S)] in fp32
    ggml_tensor * h_scan = build_rs(inp, h_states_all, hparams.n_embd_r(), S, get_h_rows);

    ggml_tensor * h_all = ggml_view_3d(ctx0, h_scan, D, T, S, D*ggml_element_size(h_scan), D*T*ggml_element_size(h_scan), 0);
    cb(h_all, "dsrn_h_all", il);

    // store the final fast state back into the cache
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            ggml_view_1d(ctx0, h_scan, D*S, D*T*S*ggml_element_size(h_scan)),
            ggml_view_1d(ctx0, h_states_all, D*S, kv_head*D*ggml_element_size(h_states_all))));

    // causal shift: x_pred[t] uses h_{t-1}; h_prev is the cached state
    ggml_tensor * h_prev = build_rs(inp, h_states_all, hparams.n_embd_r(), S);
    ggml_tensor * h_shifted;
    if (T == 1) {
        h_shifted = ggml_reshape_3d(ctx0, h_prev, D, 1, S);
    } else {
        ggml_tensor * h_all_prev = ggml_view_3d(ctx0, h_all, D, T - 1, S, h_all->nb[1], h_all->nb[2], 0);
        h_shifted = ggml_concat(ctx0, ggml_reshape_3d(ctx0, h_prev, D, 1, S), h_all_prev, 1);
    }
    cb(h_shifted, "dsrn_h_shift", il);

    // ---- surprise-gated slow state ----
    ggml_tensor * x_pred = ggml_mul_mat(ctx0, layer.dsrn_pred, h_shifted); // {D, T, S}

    ggml_tensor * diff    = ggml_sub(ctx0, x32, x_pred);
    ggml_tensor * clamped = ggml_clamp(ctx0, ggml_sqr(ctx0, diff), 0.0f, 10.0f);
    ggml_tensor * error   = ggml_mean(ctx0, clamped); // {1, T, S} (reduces over ne0)

    ggml_tensor * gate_logits = ggml_mul_mat(ctx0, layer.dsrn_gate, h_all); // {Ds, T, S}
    gate_logits = ggml_add(ctx0, gate_logits, layer.dsrn_gate_b);

    // surprise = error * softplus(lambda), broadcast over the Ds dim
    // (ggml_mul needs a full-shaped first operand; repeat the per-token error)
    ggml_tensor * surprise = ggml_mul(
            ctx0,
            ggml_repeat(ctx0, error, gate_logits),                       // {Ds, T, S}
            ggml_softplus(ctx0, layer.dsrn_lambda));                     // {Ds, 1, 1}

    gate_logits = ggml_add(ctx0, gate_logits, surprise);
    cb(gate_logits, "dsrn_gate_logits", il);

    ggml_tensor * g = ggml_sigmoid(ctx0, gate_logits);
    ggml_tensor * m = ggml_tanh(ctx0, ggml_mul_mat(ctx0, layer.dsrn_mem, h_all)); // {Ds, T, S}
    cb(g, "dsrn_g", il);
    cb(m, "dsrn_m", il);

    // ---- slow-state scan: c_t = (1 - g_t) * c_{t-1} + g_t * m_t ----
    ggml_tensor * sp_g    = ggml_softplus(ctx0, gate_logits);
    ggml_tensor * B_c     = ggml_div(ctx0, g, sp_g);
    ggml_tensor * x_c     = ggml_reshape_4d(ctx0, m, 1, Ds, T, S);
    ggml_tensor * B_c_scan = ggml_reshape_4d(ctx0, B_c, 1, Ds, T, S);

    ggml_tensor * A_c = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, Ds), -1.0f);
    ggml_tensor * C_c = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, Ds, T, S), 1.0f);

    ggml_tensor * c_states_all = mctx_cur->get_s_l(il);

    auto get_c_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
        ggml_tensor * ssm = ggml_reshape_4d(ctx, states, 1, 1, Ds, mctx_cur->get_size());
        return ggml_ssm_scan(ctx, ssm, x_c, gate_logits, A_c, B_c_scan, C_c, ids, /*K=*/1);
    };

    // c_scan: [y (Ds*T*S) | final state (Ds*S)] in fp32
    ggml_tensor * c_scan = build_rs(inp, c_states_all, hparams.n_embd_s(), S, get_c_rows);

    ggml_tensor * c_all = ggml_view_3d(ctx0, c_scan, Ds, T, S, Ds*ggml_element_size(c_scan), Ds*T*ggml_element_size(c_scan), 0);
    cb(c_all, "dsrn_c_all", il);

    // store the final slow state back into the cache
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            ggml_view_1d(ctx0, c_scan, Ds*S, Ds*T*S*ggml_element_size(c_scan)),
            ggml_view_1d(ctx0, c_states_all, Ds*S, kv_head*Ds*ggml_element_size(c_states_all))));

    // ---- memory read (purely additive residual) ----
    ggml_tensor * read = ggml_mul_mat(ctx0, layer.dsrn_read, c_all); // {D, T, S}
    ggml_tensor * x_out = ggml_add(ctx0, x32, read);
    cb(x_out, "dsrn_out", il);

    // back to the stream dtype (the reference carries the residual in bf16)
    cur = ggml_cpy(ctx0, x_out, ggml_new_tensor_3d(ctx0, stream_type, D, T, S));
    cur = ggml_reshape_2d(ctx0, cur, D, T*S);

    return cur;
}
