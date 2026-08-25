// Dump top-k logits for the last token of a fixed prompt, for parity checks
// against the HF reference. Tokenizes with add_bos=false / add_special=false
// to match the HF reference script.
#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf 'prompt' [k]  |  %s model.gguf --ids '1,2,3' [k]\n", argv[0], argv[0]); return 1; }

    const char * prompt = argv[2];
    const int k = argc > 3 ? atoi(argv[3]) : 10;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 1024;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to init context\n"); return 1; }

    // acceptance §8.2: with injectors zeroed the model must reduce to a plain
    // Qwen2 backbone (zero the dsrn_* weights in place before decoding)
    const bool zero_dsrn = getenv("ZERO_DSRN") != nullptr;
    if (zero_dsrn) {
        llama_model_tensor * t = nullptr;
        size_t idx = 0;
        while ((t = llama_model_get_tensor(model, idx)) != nullptr) {
            if (strstr(t->name, ".dsrn_") != nullptr) {
                memset(t->data, 0, t->size);
            }
            ++idx;
        }
        fprintf(stderr, "zeroed dsrn injector tensors\n");
    }

    std::vector<llama_token> tokens;
    if (strncmp(prompt, "--ids", 5) == 0) {
        // token ids given directly, comma-separated (exact-input parity tests)
        const char * p = prompt + 6;
        char * end;
        while (*p) {
            tokens.push_back((llama_token) strtol(p, &end, 10));
            if (end == p) break;
            p = end;
            while (*p == ' ' || *p == ',') p++;
        }
    } else {
        // parse_special=true so <|im_start|> etc. map to their vocab ids (like HF)
        tokens = common_tokenize(ctx, prompt, false, true);
    }
    fprintf(stderr, "n_tokens=%zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string piece = common_token_to_piece(ctx, tokens[i]);
        fprintf(stderr, "  %6d -> %s\n", tokens[i], piece.c_str());
    }

    // single decode pass over the whole prompt
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token   [i] = tokens[i];
        batch.pos     [i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i] = (i == tokens.size() - 1);
    }
    batch.n_tokens = (int) tokens.size();

    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    float * logits = llama_get_logits_ith(ctx, (int) tokens.size() - 1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // top-k
    std::vector<std::pair<float, int>> scored;
    scored.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        scored.emplace_back(logits[i], i);
    }
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });

    printf("top%d:\n", k);
    for (int i = 0; i < k; ++i) {
        const int id = scored[i].second;
        const std::string piece = common_token_to_piece(ctx, id);
        printf("  %6d  %10.4f  %s\n", id, scored[i].first, piece.c_str());
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
