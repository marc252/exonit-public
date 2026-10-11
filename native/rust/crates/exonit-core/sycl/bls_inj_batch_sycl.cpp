// PRODUCTION COPY (2026-08-06 integration, feature `sycl`) of Spike 1's
// kernel, promoted after `spikes/RESULTS_2026-08-06.md`'s real-measurement
// gate: 6 variants tried (queue mode, memory API, device-selector
// backend), all clustering at +31% to +37% median vs OpenCL on the
// production batch shape (gate: >=15%), all bit-parity-clean. The
// `baseline` variant (no `SYCL_VARIANT_*` defines -- the #else branches
// below) is what SHIPS: it measured within noise of every other variant,
// so the extra complexity of `SYCL_VARIANT_IN_ORDER`/`SYCL_VARIANT_USM`
// bought nothing and isn't worth carrying into production call sites --
// they are kept here (dead code by default) only as a documented record of
// what was tried, exactly matching the original spike file
// (`spikes/sycl-injection/bls_inj_batch_sycl.cpp`, kept as-is, historical/
// exploratory copy -- this file is the one `build.rs` actually compiles).
// Rust host module: `crates/exonit-core/src/bls/sycl_batch.rs`.
//
// SYCL transliteration of the fused f32 batched injection-recovery BLS
// kernel, verbatim source at
// crates/exonit-core/src/bls/opencl_kernels/bls_inj_batch_f32.cl (itself a
// verbatim copy of ~/exonit/worker/bls_opencl.py's
// `_BLS_BATCH_FUSED_F32_KERNEL_SRC`, lines 250-343). Spike 1 of Spec 2
// (<host>), docs/superpowers/specs/2026-08-05-native-workers-pcmarc-<host>-
// <host>-design.md.
//
// # Placeholder resolution: @NB@ / @TGSUMS@
//
// The OpenCL C source has two textual placeholders substituted BEFORE
// compilation, per n_bins, by `bls::opencl_batch::kernel_source()`
// (`.replace("@NB@", n_bins).replace("@TGSUMS@", TG_FLUX*(n_bins+1))`) --
// this bakes n_bins into the __local array sizes at COMPILE time, and is
// why `OpenClBlsBatchBackend` caches one compiled `Program` per distinct
// n_bins (`programs: Mutex<HashMap<usize, Program>>`).
//
// SYCL resolves the equivalent local memory at KERNEL SUBMISSION time
// instead, via `sycl::local_accessor<T,1>` constructed from a
// runtime-computed `sycl::range` (see `bls_inj_batch_sycl`'s `h.submit`
// lambda below: `local_accessor<float,1> tg_cumcounts(range<1>(n_bins+1),
// h)`). This needs NO per-n_bins recompilation and NO program cache --
// one compiled kernel binary handles every n_bins the caller passes,
// trading the OpenCL side's compile-cache complexity for a (cheap,
// per-submission) dynamic local-memory allocation. This is the answer to
// the task's "mateixos placeholders resolts en compile-time o runtime --
// documenta com": OpenCL = compile-time text substitution + program cache;
// SYCL = runtime local_accessor sizing, no cache needed.
//
// # Everything else: byte-for-byte port
//
// Same f32 arithmetic throughout (only the phase-fold/mean-centering that
// FEEDS this kernel is f64, done host-side by the Rust caller -- see
// `bls::opencl_batch::search_batch`, unchanged for this spike, the SYCL
// side receives the SAME already-f64-processed, f32-cast buffers).
// Same TG_FLUX=32 work-group width, same local-memory layout
// (tg_cumcounts[n_bins+1] shared + tg_sums[TG_FLUX*(n_bins+1)] one private
// row per flux slot in the work-group -- SYCL calls the OpenCL "work-group"
// concept an "nd_range work-group" too, `local_accessor` IS OpenCL __local
// under the Level Zero/SPIR-V backend).
// Same single barrier (`item.barrier(access::fence_space::local_space)` ==
// OpenCL's `barrier(CLK_LOCAL_MEM_FENCE)`), same early-return-after-barrier
// for padding work-items (`f >= n_flux`), same strict `>` tie-break in the
// duration/phase search (first (w, j) wins, matching CPU/Metal/CUDA/OpenCL).
//
// # extern "C" FFI surface
//
// `sycl_bls_inj_batch_f32` takes the SAME 9 kernel buffers (flattened to
// raw pointers + explicit dims, no `params[5]` packed array -- SYCL/C++
// has no reason to keep OpenCL's "single int buffer" convention, which
// existed there only because __kernel argument lists are otherwise
// awkward to extend) plus the dims themselves as scalar args. Returns 0 on
// success, non-zero on any SYCL exception (device init failure, kernel
// launch failure, etc.) -- errors are NOT silently swallowed into zeroed
// output, mirroring `OpenClBlsBatchBackend::search_batch`'s
// `Result<_, BlsError>` contract on the Rust side (the harness FFI wrapper
// turns a non-zero return into a Rust `Err`).

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Mirrors bls::opencl_batch::TG_FLUX exactly -- flux-slots-per-work-group,
// same fused-kernel design shared with the CUDA/OpenCL siblings.
constexpr int TG_FLUX = 32;

// Lazily-initialized GPU queue, built once and reused for every call --
// mirrors bls_opencl.py's module-level `_ctx`/`_queue` globals and
// `OpenClBlsBatchBackend`'s own `probe()`-once-then-reuse contract. Guarded
// by a mutex the same way `bls::opencl_batch::opencl_guard()` serializes
// every OpenCL submission site (2026-07-12 MLX/Metal shared-mutable-driver-
// state lesson, restated in that module's doc comment -- applies here
// identically, this is still "one process, shared GPU queue").
std::mutex g_guard;
sycl::queue *g_queue = nullptr;
std::string g_device_name;
bool g_init_attempted = false;
bool g_init_ok = false;

bool ensure_queue_locked() {
  if (g_init_attempted) {
    return g_init_ok;
  }
  g_init_attempted = true;
  try {
    // GPU-only device selection, NEVER a CPU SYCL device silently
    // substituted -- mirrors bls::opencl_batch::select_gpu_device's own
    // hard requirement (project rule 2 argument restated there).
    //
    // No `sycl::property::queue::enable_profiling()` requested anywhere in
    // this file -- SYCL profiling instrumentation on the OUT-of-order queue
    // adds event-timestamping overhead per submission; the default queue
    // already omits it, this comment just makes that an explicit choice
    // rather than an accident (2026-08-06 quiet-window iteration note --
    // "sycl::queue with in_order + no profiling" from the task instruction).
#ifdef SYCL_VARIANT_IN_ORDER
    // Iteration: in-order queue. An out-of-order SYCL queue (the default)
    // builds an implicit task-dependency GRAPH from each command's buffer/
    // accessor read-write sets and only actually orders operations that
    // touch overlapping data -- extra bookkeeping per `submit()` this
    // kernel's straight-line pipeline (upload x3 -> launch -> download x3,
    // always the same order, never re-ordered in practice) never benefits
    // from. `in_order()` tells the runtime "just run submissions in
    // program order, skip the dependency analysis" -- cheaper per-call
    // overhead at the cost of losing any out-of-order scheduling this
    // kernel was never going to use anyway.
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::gpu_selector_v);
#endif
    g_device_name = q.get_device().get_info<sycl::info::device::name>();
    g_queue = new sycl::queue(std::move(q));
    g_init_ok = true;
    std::fprintf(stderr, "bls_inj_batch_sycl: SYCL GPU queue ready on %s\n", g_device_name.c_str());
  } catch (const std::exception &error) {
    g_device_name = std::string("SYCL GPU init failed: ") + error.what();
    g_init_ok = false;
    std::fprintf(stderr, "bls_inj_batch_sycl: %s\n", g_device_name.c_str());
  }
  return g_init_ok;
}

} // namespace

extern "C" {

// Real submission smoke test -- same purpose as bls_opencl.py's
// `is_opencl_available`/`bls::opencl_batch::smoke_test`: prove the device
// can actually EXECUTE a kernel, not just that a queue could be
// constructed. Returns 1 on success, 0 on failure (opposite of the batch
// function's error convention on purpose -- this mirrors a boolean
// "is available" probe, not a fallible compute call).
int sycl_bls_smoke_test() {
  std::lock_guard<std::mutex> lock(g_guard);
  if (!ensure_queue_locked()) {
    return 0;
  }
  try {
    float result = 0.0f;
    {
      sycl::buffer<float, 1> buf(&result, sycl::range<1>(1));
      g_queue->submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) { acc[idx] = 1.0f; });
      });
    } // buffer destructor writes back to `result`.
    g_queue->wait_and_throw();
    return result == 1.0f ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "bls_inj_batch_sycl: smoke test failed: %s\n", error.what());
    return 0;
  }
}

const char *sycl_bls_last_device_name() {
  std::lock_guard<std::mutex> lock(g_guard);
  return g_device_name.empty() ? "<not yet probed>" : g_device_name.c_str();
}

#ifdef SYCL_VARIANT_USM
// Iteration: USM (Unified Shared Memory) device pointers instead of
// sycl::buffer/accessor. Same algorithm, same kernel body -- ONLY the
// host-side memory management changes: explicit sycl::malloc_device +
// queue.memcpy (async, host-to-device before the kernel, device-to-host
// after) instead of letting the SYCL runtime build an implicit task-
// dependency graph from buffer/accessor read-write sets. This removes the
// buffer/accessor bookkeeping (reference counting, access-mode tracking,
// automatic write-back on buffer destruction) at the cost of the caller
// (this function) now owning every free() explicitly -- acceptable here
// since every allocation's lifetime is exactly one function call, RAII'd
// via a small scope-guard vector below so an early exception still frees
// everything already allocated.
int sycl_bls_inj_batch_f32_usm(const int16_t *bin_idx, const float *counts, const float *flux_t, const float *totals, const int32_t *widths,
                                int32_t n_samples, int32_t n_flux, int32_t n_bins, int32_t n_per, int32_t n_dur, float *out_power, uint32_t *out_wj,
                                float *out_depth) {
  std::lock_guard<std::mutex> lock(g_guard);
  if (!ensure_queue_locked()) {
    return 1;
  }
  sycl::queue &queue = *g_queue;

  const size_t n_flux_pad = ((size_t)n_flux + TG_FLUX - 1) / TG_FLUX * TG_FLUX;
  const size_t out_len = (size_t)n_flux * (size_t)n_per;

  std::vector<void *> device_allocs;
  auto free_all = [&]() {
    for (void *p : device_allocs) {
      sycl::free(p, queue);
    }
    device_allocs.clear();
  };

  try {
    int16_t *d_bin_idx = sycl::malloc_device<int16_t>((size_t)n_per * (size_t)n_samples, queue);
    float *d_counts = sycl::malloc_device<float>((size_t)n_per * (size_t)n_bins, queue);
    float *d_flux_t = sycl::malloc_device<float>((size_t)n_samples * (size_t)n_flux, queue);
    float *d_totals = sycl::malloc_device<float>((size_t)n_flux, queue);
    int32_t *d_widths = sycl::malloc_device<int32_t>((size_t)n_dur, queue);
    float *d_out_power = sycl::malloc_device<float>(out_len, queue);
    uint32_t *d_out_wj = sycl::malloc_device<uint32_t>(out_len, queue);
    float *d_out_depth = sycl::malloc_device<float>(out_len, queue);
    device_allocs = {d_bin_idx, d_counts, d_flux_t, d_totals, d_widths, d_out_power, d_out_wj, d_out_depth};
    for (void *p : device_allocs) {
      if (p == nullptr) {
        free_all();
        std::fprintf(stderr, "bls_inj_batch_sycl: USM malloc_device failed (out of device memory?)\n");
        return 2;
      }
    }

    queue.memcpy(d_bin_idx, bin_idx, (size_t)n_per * (size_t)n_samples * sizeof(int16_t));
    queue.memcpy(d_counts, counts, (size_t)n_per * (size_t)n_bins * sizeof(float));
    queue.memcpy(d_flux_t, flux_t, (size_t)n_samples * (size_t)n_flux * sizeof(float));
    queue.memcpy(d_totals, totals, (size_t)n_flux * sizeof(float));
    queue.memcpy(d_widths, widths, (size_t)n_dur * sizeof(int32_t));

    const int32_t n_bins_k = n_bins;
    const int32_t n_dur_k = n_dur;
    const int32_t n_samples_k = n_samples;
    const int32_t n_flux_k = n_flux;
    const int32_t n_per_k = n_per;

    queue.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> tg_cumcounts(sycl::range<1>((size_t)n_bins + 1), h);
      sycl::local_accessor<float, 1> tg_sums(sycl::range<1>((size_t)TG_FLUX * ((size_t)n_bins + 1)), h);

      const sycl::range<2> global((size_t)n_flux_pad, (size_t)n_per);
      const sycl::range<2> local((size_t)TG_FLUX, 1);

      // Same identifiers as the buffer/accessor variant (a_bin_idx etc.) so
      // the KERNEL BODY below is byte-for-byte identical -- only what
      // these names are BOUND TO differs (raw USM device pointers here,
      // accessors there). Indexing syntax (`a_bin_idx[i]`) is identical
      // for both.
      const int16_t *a_bin_idx = d_bin_idx;
      const float *a_counts = d_counts;
      const float *a_flux_t = d_flux_t;
      const float *a_totals = d_totals;
      const int32_t *a_widths = d_widths;
      float *a_out_power = d_out_power;
      uint32_t *a_out_wj = d_out_wj;
      float *a_out_depth = d_out_depth;

      h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
        const int32_t f = (int32_t)item.get_global_id(0);
        const int32_t p = (int32_t)item.get_global_id(1);
        const int32_t lid = (int32_t)item.get_local_id(0);

        if (lid == 0) {
          float c = 0.0f;
          tg_cumcounts[0] = 0.0f;
          const size_t crow = (size_t)p * (size_t)n_bins_k;
          for (int32_t k = 0; k < n_bins_k; k++) {
            c += a_counts[crow + (size_t)k];
            tg_cumcounts[k + 1] = c;
          }
        }
        item.barrier(sycl::access::fence_space::local_space);

        if (f >= n_flux_k) {
          return;
        }

        const size_t sums_base = (size_t)lid * ((size_t)n_bins_k + 1);
        for (int32_t k = 0; k <= n_bins_k; k++) {
          tg_sums[sums_base + (size_t)k] = 0.0f;
        }

        const size_t brow = (size_t)p * (size_t)n_samples_k;
        for (int32_t i = 0; i < n_samples_k; i++) {
          const int32_t bin = (int32_t)a_bin_idx[brow + (size_t)i];
          tg_sums[sums_base + (size_t)(bin + 1)] += a_flux_t[(size_t)i * (size_t)n_flux_k + (size_t)f];
        }
        for (int32_t k = 1; k <= n_bins_k; k++) {
          tg_sums[sums_base + (size_t)k] += tg_sums[sums_base + (size_t)(k - 1)];
        }

        const float total = a_totals[f];
        const float n_float = (float)n_samples_k;

        float best_sr = -1e30f;
        int32_t best_w = 0;
        int32_t best_j = 0;
        float best_dep = 0.0f;

        for (int32_t di = 0; di < n_dur_k; di++) {
          const int32_t w = a_widths[di];
          for (int32_t j = 0; j < n_bins_k; j++) {
            const int32_t e = j + w;
            float wc, ws;
            if (e <= n_bins_k) {
              wc = tg_cumcounts[e] - tg_cumcounts[j];
              ws = tg_sums[sums_base + (size_t)e] - tg_sums[sums_base + (size_t)j];
            } else {
              const int32_t e2 = e - n_bins_k;
              wc = (tg_cumcounts[n_bins_k] - tg_cumcounts[j]) + tg_cumcounts[e2];
              ws = (tg_sums[sums_base + (size_t)n_bins_k] - tg_sums[sums_base + (size_t)j]) + tg_sums[sums_base + (size_t)e2];
            }
            const float oc = n_float - wc;
            if (wc < 3.0f || oc < 3.0f) {
              continue;
            }
            const float mean_in = ws / wc;
            const float mean_out = (total - ws) / oc;
            const float depth = mean_out - mean_in;
            if (depth > 0.0f) {
              const float sr = depth * depth * wc * oc / (wc + oc);
              if (sr > best_sr) {
                best_sr = sr;
                best_w = w;
                best_j = j;
                best_dep = depth;
              }
            }
          }
        }

        const size_t o = (size_t)f * (size_t)n_per_k + (size_t)p;
        a_out_power[o] = best_sr;
        a_out_wj[o] = ((uint32_t)best_w << 16) | ((uint32_t)best_j & 0xFFFFu);
        a_out_depth[o] = best_dep;
      });
    });
    queue.wait_and_throw();

    queue.memcpy(out_power, d_out_power, out_len * sizeof(float));
    queue.memcpy(out_wj, d_out_wj, out_len * sizeof(uint32_t));
    queue.memcpy(out_depth, d_out_depth, out_len * sizeof(float));
    queue.wait_and_throw();
    free_all();
  } catch (const std::exception &error) {
    free_all();
    std::fprintf(stderr, "bls_inj_batch_sycl: USM search failed: %s\n", error.what());
    return 2;
  }
  return 0;
}
#else
// Line-for-line port of `bls_inj_batch_f32`'s host-visible contract:
// buffers laid out EXACTLY as `bls::opencl_batch::search_batch` builds them
// (transposed fluxT, i16 bin_idx, packed dims -- here flattened to scalar
// args, see file header). Global range = (n_flux_pad, n_per), local range =
// (TG_FLUX, 1), n_flux_pad rounded up to the next TG_FLUX multiple exactly
// as the Rust caller already computes it (this function trusts the caller
// on n_flux_pad sizing of the padding work-items; out-of-range `f` lanes
// return before writing, identical to the OpenCL kernel body).
int sycl_bls_inj_batch_f32_buffers(const int16_t *bin_idx,   // (n_per, N)
                            const float *counts,      // (n_per, n_bins)
                            const float *flux_t,      // (N, n_flux)
                            const float *totals,      // (n_flux,)
                            const int32_t *widths,     // (n_dur,)
                            int32_t n_samples, int32_t n_flux, int32_t n_bins, int32_t n_per, int32_t n_dur,
                            float *out_power,   // (n_flux, n_per)
                            uint32_t *out_wj,   // (n_flux, n_per)
                            float *out_depth) { // (n_flux, n_per)
  std::lock_guard<std::mutex> lock(g_guard);
  if (!ensure_queue_locked()) {
    return 1;
  }
  sycl::queue &queue = *g_queue;

  try {
    const size_t n_flux_pad = ((size_t)n_flux + TG_FLUX - 1) / TG_FLUX * TG_FLUX;
    const size_t out_len = (size_t)n_flux * (size_t)n_per;

    sycl::buffer<int16_t, 1> buf_bin_idx(bin_idx, sycl::range<1>((size_t)n_per * (size_t)n_samples));
    sycl::buffer<float, 1> buf_counts(counts, sycl::range<1>((size_t)n_per * (size_t)n_bins));
    sycl::buffer<float, 1> buf_flux_t(flux_t, sycl::range<1>((size_t)n_samples * (size_t)n_flux));
    sycl::buffer<float, 1> buf_totals(totals, sycl::range<1>((size_t)n_flux));
    sycl::buffer<int32_t, 1> buf_widths(widths, sycl::range<1>((size_t)n_dur));
    sycl::buffer<float, 1> buf_out_power(out_power, sycl::range<1>(out_len));
    sycl::buffer<uint32_t, 1> buf_out_wj(out_wj, sycl::range<1>(out_len));
    sycl::buffer<float, 1> buf_out_depth(out_depth, sycl::range<1>(out_len));

    queue.submit([&](sycl::handler &h) {
      auto a_bin_idx = buf_bin_idx.get_access<sycl::access::mode::read>(h);
      auto a_counts = buf_counts.get_access<sycl::access::mode::read>(h);
      auto a_flux_t = buf_flux_t.get_access<sycl::access::mode::read>(h);
      auto a_totals = buf_totals.get_access<sycl::access::mode::read>(h);
      auto a_widths = buf_widths.get_access<sycl::access::mode::read>(h);
      auto a_out_power = buf_out_power.get_access<sycl::access::mode::write>(h);
      auto a_out_wj = buf_out_wj.get_access<sycl::access::mode::write>(h);
      auto a_out_depth = buf_out_depth.get_access<sycl::access::mode::write>(h);

      // Runtime-sized local memory -- see file header, "Placeholder
      // resolution". tg_cumcounts: n_bins+1 shared floats. tg_sums:
      // TG_FLUX*(n_bins+1), one private row per flux slot in the
      // work-group -- identical formula to
      // bls::opencl_batch::local_mem_bytes(n_bins)/4.
      sycl::local_accessor<float, 1> tg_cumcounts(sycl::range<1>((size_t)n_bins + 1), h);
      sycl::local_accessor<float, 1> tg_sums(sycl::range<1>((size_t)TG_FLUX * ((size_t)n_bins + 1)), h);

      const sycl::range<2> global((size_t)n_flux_pad, (size_t)n_per);
      const sycl::range<2> local((size_t)TG_FLUX, 1);

      const int32_t n_bins_k = n_bins;
      const int32_t n_dur_k = n_dur;
      const int32_t n_samples_k = n_samples;
      const int32_t n_flux_k = n_flux;
      const int32_t n_per_k = n_per;

      h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
        const int32_t f = (int32_t)item.get_global_id(0);   // flux slot (padded to TG_FLUX)
        const int32_t p = (int32_t)item.get_global_id(1);   // period index within this chunk
        const int32_t lid = (int32_t)item.get_local_id(0);

        // --- shared per-period cumulative counts (exact: integers in f32) ---
        if (lid == 0) {
          float c = 0.0f;
          tg_cumcounts[0] = 0.0f;
          const size_t crow = (size_t)p * (size_t)n_bins_k;
          for (int32_t k = 0; k < n_bins_k; k++) {
            c += a_counts[crow + (size_t)k];
            tg_cumcounts[k + 1] = c;
          }
        }
        item.barrier(sycl::access::fence_space::local_space);

        if (f >= n_flux_k) {
          return; // padding work-items: after the only barrier, same as the .cl source.
        }

        // --- private per-bin flux sums (sample order == numba/OpenCL order) ---
        const size_t sums_base = (size_t)lid * ((size_t)n_bins_k + 1);
        for (int32_t k = 0; k <= n_bins_k; k++) {
          tg_sums[sums_base + (size_t)k] = 0.0f;
        }

        const size_t brow = (size_t)p * (size_t)n_samples_k;
        for (int32_t i = 0; i < n_samples_k; i++) {
          const int32_t bin = (int32_t)a_bin_idx[brow + (size_t)i];
          tg_sums[sums_base + (size_t)(bin + 1)] += a_flux_t[(size_t)i * (size_t)n_flux_k + (size_t)f];
        }
        // in-place prefix sum: tg_sums[sums_base+k] = sum of bins [0, k)
        for (int32_t k = 1; k <= n_bins_k; k++) {
          tg_sums[sums_base + (size_t)k] += tg_sums[sums_base + (size_t)(k - 1)];
        }

        const float total = a_totals[f];
        const float n_float = (float)n_samples_k;

        float best_sr = -1e30f;
        int32_t best_w = 0;
        int32_t best_j = 0;
        float best_dep = 0.0f;

        for (int32_t di = 0; di < n_dur_k; di++) {
          const int32_t w = a_widths[di];
          for (int32_t j = 0; j < n_bins_k; j++) {
            const int32_t e = j + w; // window [j, e), wraps past n_bins
            float wc, ws;
            if (e <= n_bins_k) {
              wc = tg_cumcounts[e] - tg_cumcounts[j];
              ws = tg_sums[sums_base + (size_t)e] - tg_sums[sums_base + (size_t)j];
            } else {
              const int32_t e2 = e - n_bins_k;
              wc = (tg_cumcounts[n_bins_k] - tg_cumcounts[j]) + tg_cumcounts[e2];
              ws = (tg_sums[sums_base + (size_t)n_bins_k] - tg_sums[sums_base + (size_t)j]) + tg_sums[sums_base + (size_t)e2];
            }
            const float oc = n_float - wc;
            if (wc < 3.0f || oc < 3.0f) {
              continue;
            }
            const float mean_in = ws / wc;
            const float mean_out = (total - ws) / oc;
            const float depth = mean_out - mean_in;
            if (depth > 0.0f) {
              const float sr = depth * depth * wc * oc / (wc + oc);
              if (sr > best_sr) { // strict '>': first (w, j) wins ties
                best_sr = sr;
                best_w = w;
                best_j = j;
                best_dep = depth;
              }
            }
          }
        }

        const size_t o = (size_t)f * (size_t)n_per_k + (size_t)p;
        a_out_power[o] = best_sr;
        a_out_wj[o] = ((uint32_t)best_w << 16) | ((uint32_t)best_j & 0xFFFFu);
        a_out_depth[o] = best_dep;
      });
    });
    queue.wait_and_throw();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "bls_inj_batch_sycl: search failed: %s\n", error.what());
    return 2;
  }
  return 0;
}
#endif // SYCL_VARIANT_USM

// Exported symbol the Rust harness links against (`#[link(name =
// "sycl_bls_inj")]` + `sycl_bls_inj_batch_f32` in
// harness/src/ffi.rs) -- stable across every iteration variant in
// RESULTS_2026-08-06.md, so the harness never needs to change when
// build.sh's SYCL_VARIANT_* defines pick a different implementation.
int sycl_bls_inj_batch_f32(const int16_t *bin_idx, const float *counts, const float *flux_t, const float *totals, const int32_t *widths,
                            int32_t n_samples, int32_t n_flux, int32_t n_bins, int32_t n_per, int32_t n_dur, float *out_power, uint32_t *out_wj,
                            float *out_depth) {
#ifdef SYCL_VARIANT_USM
  return sycl_bls_inj_batch_f32_usm(bin_idx, counts, flux_t, totals, widths, n_samples, n_flux, n_bins, n_per, n_dur, out_power, out_wj, out_depth);
#else
  return sycl_bls_inj_batch_f32_buffers(bin_idx, counts, flux_t, totals, widths, n_samples, n_flux, n_bins, n_per, n_dur, out_power, out_wj, out_depth);
#endif
}

} // extern "C"
