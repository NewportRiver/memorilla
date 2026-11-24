#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Args {
    std::string model, ids, embd, out, mode = "token";
    int ctx = 256, steps = 4, threads = 1, external_index = -1;
};

static void die(const std::string & s) { throw std::runtime_error(s); }
static int arg_int(int argc, char ** argv, int & i) { if (++i >= argc) die("missing argument"); return std::stoi(argv[i]); }
static std::string arg_str(int argc, char ** argv, int & i) { if (++i >= argc) die("missing argument"); return argv[i]; }

static Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string x = argv[i];
        if (x == "--model") a.model = arg_str(argc, argv, i);
        else if (x == "--ids") a.ids = arg_str(argc, argv, i);
        else if (x == "--embd") a.embd = arg_str(argc, argv, i);
        else if (x == "--out") a.out = arg_str(argc, argv, i);
        else if (x == "--mode") a.mode = arg_str(argc, argv, i);
        else if (x == "--ctx") a.ctx = arg_int(argc, argv, i);
        else if (x == "--steps") a.steps = arg_int(argc, argv, i);
        else if (x == "--threads") a.threads = arg_int(argc, argv, i);
        else if (x == "--external-index") a.external_index = arg_int(argc, argv, i);
        else if (x == "--help") {
            std::cout << "--model GGUF --ids ids.bin --embd rows.f32 --out result.bin --mode token|embd|mixed --ctx N --steps N --external-index N\n";
            std::exit(0);
        } else die("unknown argument: " + x);
    }
    if (a.model.empty() || a.ids.empty() || a.embd.empty() || a.out.empty()) die("model, ids, embd, and out are required");
    if (a.mode != "token" && a.mode != "embd" && a.mode != "mixed") die("invalid mode");
    return a;
}

static std::vector<int32_t> read_ids(const std::string & path) {
    std::ifstream f(path, std::ios::binary); if (!f) die("cannot open ids");
    f.seekg(0, std::ios::end); const auto n = f.tellg(); f.seekg(0);
    if (n <= 0 || n % sizeof(int32_t)) die("invalid ids file");
    std::vector<int32_t> v(static_cast<size_t>(n) / sizeof(int32_t));
    f.read(reinterpret_cast<char *>(v.data()), n); return v;
}
static std::vector<float> read_floats(const std::string & path) {
    std::ifstream f(path, std::ios::binary); if (!f) die("cannot open embeddings");
    f.seekg(0, std::ios::end); const auto n = f.tellg(); f.seekg(0);
    if (n <= 0 || n % sizeof(float)) die("invalid embeddings file");
    std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
    f.read(reinterpret_cast<char *>(v.data()), n); return v;
}
static void write_floats(const std::string & path, const std::vector<float> & v) {
    std::ofstream f(path, std::ios::binary); if (!f) die("cannot write output");
    f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
}

static int32_t greedy(const float * logits, int32_t n_vocab) {
    return static_cast<int32_t>(std::max_element(logits, logits + n_vocab) - logits);
}

int main(int argc, char ** argv) {
    try {
        const Args a = parse(argc, argv);
        const auto ids = read_ids(a.ids);
        const auto embd = read_floats(a.embd);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
        if (!model) die("model load failed");
        const int32_t n_embd = llama_model_n_embd_inp(model);
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (n_embd <= 0 || n_vocab <= 0) die("invalid model dimensions");
        if (embd.size() < ids.size() * static_cast<size_t>(n_embd)) die("embedding rows are too short");
        if (a.ctx < static_cast<int>(ids.size()) + a.steps) die("context overflow before native submission");
        if (a.mode == "mixed" && (a.external_index < 0 || a.external_index >= static_cast<int>(ids.size()))) die("external index out of range");
        for (size_t i = 0; i < ids.size() * static_cast<size_t>(n_embd); ++i) {
            if (!std::isfinite(embd[i])) die("embedding contains non-finite value");
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = static_cast<uint32_t>(a.ctx);
        cp.n_batch = static_cast<uint32_t>(std::max<size_t>(ids.size(), 32));
        cp.n_ubatch = cp.n_batch;
        cp.n_seq_max = 1;
        cp.no_perf = false;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) die("context creation failed");
        llama_set_n_threads(ctx, a.threads, a.threads);
        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        if (!batch) die("batch creation failed");

        std::vector<float> final_logits;
        std::vector<int32_t> generated;
        const auto submit = [&](int32_t id, const float * row, llama_pos pos, bool output) {
            llama_batch_ext_clear(batch);
            int32_t idx;
            if (row) idx = llama_batch_ext_add_embd(batch, 0, {row, 1, static_cast<size_t>(n_embd)});
            else idx = llama_batch_ext_add_token(batch, 0, id);
            if (idx < 0) die("batch add failed: " + std::to_string(idx));
            if (!llama_batch_ext_set_pos(batch, idx, &pos)) die("position set failed");
            if (!llama_batch_ext_set_output_logits(batch, idx, output)) die("logit flag failed");
            const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
            if (rc != 0) die("llama_process failed: " + std::to_string(rc));
        };

        for (size_t i = 0; i < ids.size(); ++i) {
            const bool use_embd = a.mode == "embd" || (a.mode == "mixed" && static_cast<int>(i) == a.external_index);
            submit(ids[i], use_embd ? embd.data() + i * static_cast<size_t>(n_embd) : nullptr, static_cast<llama_pos>(i), i + 1 == ids.size());
        }
        const float * logits = llama_get_logits(ctx);
        if (!logits) die("logits unavailable");
        final_logits.assign(logits, logits + n_vocab);
        for (int step = 0; step < a.steps; ++step) {
            const int32_t next = greedy(final_logits.data(), n_vocab);
            generated.push_back(next);
            const llama_pos pos = static_cast<llama_pos>(ids.size() + step);
            submit(next, nullptr, pos, true);
            logits = llama_get_logits(ctx); if (!logits) die("continuation logits unavailable");
            final_logits.assign(logits, logits + n_vocab);
        }
        write_floats(a.out + ".logits.f32", final_logits);
        std::ofstream meta(a.out); if (!meta) die("cannot write metadata");
        meta << "{\"mode\":\"" << a.mode << "\",\"n_embd\":" << n_embd << ",\"n_vocab\":" << n_vocab << ",\"input_tokens\":" << ids.size() << ",\"ctx\":" << a.ctx << ",\"generated\":[";
        for (size_t i = 0; i < generated.size(); ++i) { if (i) meta << ','; meta << generated[i]; }
        meta << "]}\n";
        llama_batch_ext_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    } catch (const std::exception & e) { std::cerr << "HARNESS_ERROR: " << e.what() << '\n'; return 2; }
}
