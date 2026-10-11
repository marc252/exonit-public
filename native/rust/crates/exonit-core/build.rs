//! Compiles the two external-toolchain kernels this crate optionally
//! embeds: SYCL (feature `sycl`, icpx) and ISPC (feature `ispc`, ispc).
//! Both do NOTHING when their feature is off — no external toolchain is
//! ever invoked, and the crate builds exactly as it did before either
//! feature existed (mirrors `cudarc`'s `dynamic-loading` posture: compiling
//! WITHOUT the hardware/toolchain present must always work).
//!
//! Neither kernel is a crates.io dependency (unlike `ocl`/`cudarc`) because
//! neither icpx nor ispc has a Rust binding crate at all — both are
//! external C/SPMD compilers this script shells out to directly, the same
//! way a `cc`-crate build script would, just without `cc`'s own bundled
//! compiler-detection logic (icpx/ispc are found via `ONEAPI_ROOT`/
//! `ISPC_ROOT` env vars or documented default install paths, see each
//! `build_*` function below).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    build_sycl();
    build_ispc();
}

/// 2026-08-06 Spec 2 integration (`spikes/RESULTS_2026-08-06.md`'s SYCL
/// gate: +31-37% vs OpenCL, ships). Compiles
/// `sycl/bls_inj_batch_sycl.cpp` with icpx into a shared library
/// (`libsycl_bls_inj.so`) and links it into whatever binary this crate
/// ends up in — see `bls::sycl_batch`'s module doc comment for the Rust
/// FFI side.
///
/// Shared (not static) because a SYCL translation unit pulls in the SYCL
/// runtime's OWN shared libraries transitively (`libsycl.so`,
/// `libze_loader.so` via Level Zero, `libsvml.so`/`libirng.so`/
/// `libimf.so`/`libintlc.so.5` — Intel's short-vector-math runtime,
/// discovered running `cargo test` for the first time: `ldd` reported all
/// five "not found" even with rpath set on the FINAL Rust binary, because
/// `DT_RUNPATH` is NOT transitive across shared-library boundaries — it
/// only governs resolving ITS OWN object's direct dependencies, not a
/// dependency-of-a-dependency's. Fixed by ALSO passing `-Wl,-rpath,...`
/// directly to the `icpx` invocation below, baking the oneAPI lib dirs
/// into `libsycl_bls_inj.so` itself) — there is no static-link story for
/// the SYCL runtime itself, so producing our OWN shared lib is not adding
/// an extra dependency class, just one more `.so` alongside ones the SYCL
/// runtime already requires.
///
/// **This does NOT fully eliminate `LD_LIBRARY_PATH` at deploy time**,
/// unlike ISPC below: the SYCL Unified Runtime loads its actual device
/// adapter plugins (`libur_adapter_level_zero.so.0`/
/// `libur_adapter_opencl.so.0`, both living in `compiler/latest/lib`) via
/// an internal `dlopen()` call made from WITHIN `libsycl.so`'s own runtime
/// code — and `dlopen()` calls issued by application/library code do NOT
/// consult that library's own `DT_RUNPATH` (only `LD_LIBRARY_PATH`/the
/// system default paths/`ld.so.cache` do, or the older, now-deprecated
/// `DT_RPATH` — confirmed empirically: `sycl_bls_smoke_test()` failed with
/// "No device of requested type gpu available" despite `ldd` on the
/// binary itself showing every direct dependency resolved, and only
/// started finding the Arc once `LD_LIBRARY_PATH` covered
/// `compiler/latest/lib` + `umf/latest/lib` + `tcm/latest/lib` +
/// `tbb/latest/lib` — same four dirs `spikes/sycl-injection/
/// oneapi-env.sh` already documented needing). **Whatever launches the
/// production worker binary on <host> (farmctl/systemd unit, once a SYCL-
/// enabled worker instance is actually deployed) MUST set this
/// `LD_LIBRARY_PATH` — this is a real, unavoidable deployment requirement
/// of SYCL's plugin architecture, not something `build.rs`/rpath tricks
/// can paper over.** `rpath` entries (below) still matter for the parts
/// they DO cover (this crate's own `libsycl_bls_inj.so`'s location, and
/// every direct `DT_NEEDED` dependency of both it and the final binary).
fn build_sycl() {
    println!("cargo:rerun-if-env-changed=ONEAPI_ROOT");
    if env::var_os("CARGO_FEATURE_SYCL").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let cpp_source = manifest_dir.join("sycl/bls_inj_batch_sycl.cpp");
    println!("cargo:rerun-if-changed={}", cpp_source.display());

    let oneapi_root = env::var("ONEAPI_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env::var("HOME").expect("HOME (needed to default ONEAPI_ROOT to ~/intel/oneapi)")).join("intel/oneapi"));
    let icpx = oneapi_root.join("compiler/latest/bin/icpx");
    if !icpx.exists() {
        panic!(
            "exonit-core build.rs: feature `sycl` is enabled but icpx was not found at {}\n\
             -- install the Intel oneAPI DPC++/C++ Compiler (see spikes/README.md, \"Toolchains \
             installed\" section, for the exact offline-installer steps used on <host>) and \
             either export ONEAPI_ROOT=<install dir> or install to the default ~/intel/oneapi.",
            icpx.display()
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let lib_path = out_dir.join("libsycl_bls_inj.so");

    let oneapi_lib_dirs = [
        oneapi_root.join("compiler/latest/lib"),
        oneapi_root.join("umf/latest/lib"),
        oneapi_root.join("tcm/latest/lib"),
        oneapi_root.join("tbb/latest/lib"),
    ];

    // nice -n 10: <host> is the production server (systemd exonit.service)
    // even when this crate is being (re)built there with a live run —
    // never claim more build parallelism than one nice'd icpx process
    // (project rule 2).
    //
    // `-Wl,-rpath,<oneapi lib dir>` PASSED TO ICPX ITSELF (not just to the
    // final Rust binary further down) — discovered the hard way running
    // `cargo test`: `libsycl_bls_inj.so` itself transitively depends on
    // `libsycl.so.9`/`libsvml.so`/`libirng.so`/`libimf.so`/
    // `libintlc.so.5` (icpx's own runtime + Intel's short-vector-math
    // library), and `DT_RUNPATH` is NOT transitive across shared library
    // boundaries — the FINAL BINARY's own runpath (set via
    // `cargo:rustc-link-arg` below) only governs resolving the binary's
    // DIRECT dependencies, not `libsycl_bls_inj.so`'s OWN dependencies.
    // Without an rpath baked into `libsycl_bls_inj.so` itself, `ldd`
    // reports every one of those five as "not found" the moment the .so
    // needs LOADING (a `cargo check`, which never links+runs, does not
    // catch this — only `cargo test`/`cargo run` do).
    let mut icpx_args: Vec<String> = vec!["-fsycl".into(), "-fsycl-targets=spir64".into(), "-O3".into(), "-fPIC".into(), "-shared".into()];
    for dir in &oneapi_lib_dirs {
        icpx_args.push(format!("-Wl,-rpath,{}", dir.display()));
    }
    let status = Command::new("nice")
        .args(["-n", "10"])
        .arg(&icpx)
        .args(&icpx_args)
        .arg("-o")
        .arg(&lib_path)
        .arg(&cpp_source)
        .status()
        .unwrap_or_else(|error| panic!("exonit-core build.rs: failed to invoke `nice icpx` at {}: {error}", icpx.display()));
    if !status.success() {
        panic!("exonit-core build.rs: icpx failed to compile {} (exit status: {status})", cpp_source.display());
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=sycl_bls_inj");
    // Also set on the FINAL binary (belt-and-suspenders, and needed for
    // `out_dir` itself — `libsycl_bls_inj.so`'s own location — since that
    // one genuinely IS a direct dependency of the final binary).
    for rpath in std::iter::once(out_dir.clone()).chain(oneapi_lib_dirs.iter().cloned()) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath.display());
    }
}

/// 2026-08-06 Spec 2 integration (`spikes/RESULTS_2026-08-06.md`'s ISPC
/// gate: +55-235% vs Rust autovec, ships). Compiles
/// `ispc/bls_fold_ispc.ispc` with ispc into a plain object file, archives
/// it into a static lib (`libispc_bls_fold.a`), and links it STATICALLY
/// into whatever binary this crate ends up in.
///
/// Static (unlike the SYCL `.so` above): ISPC's `avx2-i32x8` codegen has
/// NO runtime dependency beyond `libm` (`floor()`), already linked into
/// every Rust binary via libc/std — there is no ISPC "runtime" shared
/// library the way SYCL has one, so static linking is strictly simpler
/// (self-contained binary, no rpath needed for this half) and matches
/// project rule 4's "easy install" preference wherever the toolchain
/// allows it. No `launch[]`/tasking in `bls_fold_ispc.ispc` (see that
/// file's header comment), so `tasksys.cpp`/ISPC's task-parallel runtime
/// object is deliberately NOT linked here either — this build produces the
/// smallest artifact that actually runs the kernel.
fn build_ispc() {
    println!("cargo:rerun-if-env-changed=ISPC_ROOT");
    if env::var_os("CARGO_FEATURE_ISPC").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let ispc_source = manifest_dir.join("ispc/bls_fold_ispc.ispc");
    println!("cargo:rerun-if-changed={}", ispc_source.display());

    // `HOME` is set on <host> too (confirmed 2026-08-07 -- some part of its
    // toolchain stack, likely git-for-windows, already exports it equal to
    // `%USERPROFILE%`), so the same `~/toolchains/ispc/current` default
    // layout used on <host> works verbatim on Windows -- no `USERPROFILE`
    // fallback needed here. `bin/ispc` -> `bin/ispc.exe` is the only path
    // difference the two platforms' release archives have.
    let ispc_root = env::var("ISPC_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env::var("HOME").expect("HOME (needed to default ISPC_ROOT to ~/toolchains/ispc/current)")).join("toolchains/ispc/current"));
    let ispc_bin_relative = if cfg!(windows) { "bin/ispc.exe" } else { "bin/ispc" };
    let ispc_bin = ispc_root.join(ispc_bin_relative);
    if !ispc_bin.exists() {
        panic!(
            "exonit-core build.rs: feature `ispc` is enabled but ispc was not found at {}\n\
             -- install ISPC (see spikes/README.md, \"Toolchains installed\" section, for the \
             exact GitHub-release steps used on <host> -- docs/RUNBOOK_PCMARC.md documents the \
             Windows equivalent) and either export ISPC_ROOT=<install dir> or install to the \
             default ~/toolchains/ispc/current.",
            ispc_bin.display()
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    // ispc auto-detects the object format for the OS it's actually running
    // on (COFF `.obj` natively on Windows, ELF `.o` on Linux) -- no
    // `--target-os` override needed, matching how the compiler is invoked
    // identically on both platforms below.
    let obj_extension = if cfg!(windows) { "obj" } else { "o" };
    let obj_path = out_dir.join(format!("bls_fold_ispc.{obj_extension}"));

    // `nice -n 10` (host-courtesy, same rationale as build_sycl() above) is
    // a Unix-only binary -- Windows has no equivalent built-in and <host>'s
    // project rule 2 host-courtesy concern doesn't apply to <host>'s dev checkout
    // (not a shared production host the way <host> is), so just invoke
    // ispc directly there.
    let mut command = if cfg!(unix) {
        let mut command = Command::new("nice");
        command.args(["-n", "10"]).arg(&ispc_bin);
        command
    } else {
        Command::new(&ispc_bin)
    };
    command.args(["-O3", "--target=avx2-i32x8", "--pic"]).arg(&ispc_source).arg("-o").arg(&obj_path);
    let status = command.status().unwrap_or_else(|error| panic!("exonit-core build.rs: failed to invoke ispc at {}: {error}", ispc_bin.display()));
    if !status.success() {
        panic!("exonit-core build.rs: ispc failed to compile {} (exit status: {status})", ispc_source.display());
    }

    if cfg!(windows) {
        // MSVC: `ispc_fold.rs`'s FFI extern block carries
        // `#[link(name = "ispc_bls_fold")]` (source-level, NOT a build.rs
        // directive) -- rustc bakes that native-lib requirement into
        // exonit-core's own compiled rlib metadata, so every downstream
        // binary's link line unconditionally demands `ispc_bls_fold.lib`
        // regardless of what this build script emits. Passing the raw
        // `.obj` straight to the linker via `cargo:rustc-link-arg` (what
        // an earlier version of this function did) does NOT satisfy that:
        // `link.exe` still separately searches for `ispc_bls_fold.lib` on
        // its `-l`-equivalent path and fails with `LNK1181: cannot open
        // input file 'ispc_bls_fold.lib'` (found running this for real on
        // <host>, 2026-08-07) even though the .obj it needed was already
        // on the link line by full path. So an actual `ispc_bls_fold.lib`
        // has to exist, matching the Unix `ar`-archive path's spirit (a
        // real named static lib, not just a bag of link args) -- produced
        // here via MSVC's own librarian, `lib.exe` (the standard tool for
        // this on Windows; GNU `ar` isn't part of the MSVC toolchain and
        // an `ar`-built `.a` isn't guaranteed to be a COFF archive
        // `link.exe` accepts). Located via `cc::windows_registry::find`
        // (registry/vswhere-based, like `cc`/`cmake` do it) rather than a
        // hardcoded `Program Files\...\VC\Tools\MSVC\<version>\...` path,
        // which would break on the next VS Build Tools update.
        let target = env::var("TARGET").expect("TARGET (needed to locate lib.exe via cc::windows_registry::find)");
        let lib_path = out_dir.join("ispc_bls_fold.lib");
        let mut lib_exe = cc::windows_registry::find(&target, "lib.exe").unwrap_or_else(|| {
            panic!(
                "exonit-core build.rs: feature `ispc` is enabled but lib.exe (MSVC librarian) was \
                 not found via cc::windows_registry::find for target {target} -- install VS Build \
                 Tools (see docs/RUNBOOK_PCMARC.md \"Toolchain\" section)."
            )
        });
        let lib_status = lib_exe
            .arg(format!("/OUT:{}", lib_path.display()))
            .arg(&obj_path)
            .status()
            .unwrap_or_else(|error| panic!("exonit-core build.rs: failed to invoke lib.exe: {error}"));
        if !lib_status.success() {
            panic!("exonit-core build.rs: lib.exe failed to archive {} (exit status: {lib_status})", obj_path.display());
        }
    } else {
        let lib_path = out_dir.join("libispc_bls_fold.a");
        let ar_status = Command::new("ar")
            .arg("rcs")
            .arg(&lib_path)
            .arg(&obj_path)
            .status()
            .unwrap_or_else(|error| panic!("exonit-core build.rs: failed to invoke `ar`: {error}"));
        if !ar_status.success() {
            panic!("exonit-core build.rs: ar failed to archive {} (exit status: {ar_status})", obj_path.display());
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // `kind=static` here is NOT redundant with `ispc_fold.rs`'s
    // `#[link(name = "ispc_bls_fold")]` FFI attribute, which carries no
    // `kind` of its own -- discovered the hard way on Windows (2026-08-07):
    // without this directive, an unqualified `#[link(name = "...")]`
    // defaults to `kind = "dylib"` on the MSVC target, so rustc compiles
    // exonit-core's calls into the ISPC kernel as `dllimport`-style
    // indirect calls (`__imp_ispc_bls_fold_batch`), which nothing provides
    // -- `LNK2019: unresolved external symbol __imp_ispc_bls_fold_batch`,
    // even though `ispc_bls_fold.lib` itself was found and linked fine.
    // This `cargo:rustc-link-lib=static=...` output is what actually
    // supplies `kind = "static"` for that attribute at exonit-core's OWN
    // compile time (Cargo feeds a build script's native-lib directives to
    // rustc as `-l` flags when compiling THAT crate, filling in what the
    // source attribute left unspecified) -- on Unix this already worked
    // for the same reason, it just never got exercised as an "or the
    // dllimport codegen path fires" case the way ELF static linking does.
    println!("cargo:rustc-link-lib=static=ispc_bls_fold");
}
