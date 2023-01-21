use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn target_env_is(name: &str) -> bool {
    match env::var_os("CARGO_CFG_TARGET_ENV") {
        Some(s) => s == name,
        None => false,
    }
}

fn target_os_is(name: &str) -> bool {
    match env::var_os("CARGO_CFG_TARGET_OS") {
        Some(s) => s == name,
        None => false,
    }
}

pub fn llvm_config(path: &Path, arg: &str) -> PathBuf {
    let stdout = Command::new(path)
        .arg(arg)
        .arg("--link-static")
        .output()
        .unwrap()
        .stdout;
    PathBuf::from(String::from_utf8(stdout).unwrap().trim())
}

pub fn setup_llvm_build<P: AsRef<Path>>(libs: &[&str], include_dirs: &[P]) {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let base_dir = crate_dir.parent().unwrap().parent().unwrap();
    let circt_dir = base_dir.join("circt");
    let circt_build_dir = circt_dir.join("build");
    let llvm_dir = circt_dir.join("llvm");
    let llvm_build_dir = llvm_dir.join("build");

    let llvm_config_path = llvm_build_dir.join("bin").join("llvm-config");
    let llvm_lib_dir = llvm_config(&llvm_config_path, "--libdir");

    println!("cargo:rustc-link-search=native={}", llvm_lib_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        circt_build_dir.join("lib").display()
    );

    for name in libs {
        println!("cargo:rustc-link-lib=static={name}");
    }

    // Get the library that must be linked for C++, if any.
    // Source: https://gitlab.com/taricorp/llvm-sys.rs/-/blob/main/build.rs
    let system_libcpp = if target_env_is("msvc") {
        None
    } else if target_os_is("macos") || target_os_is("freebsd") {
        Some("c++")
    } else {
        Some("stdc++")
    };
    if let Some(system_libcpp) = system_libcpp {
        println!("cargo:rustc-link-lib=dylib={system_libcpp}");
    }

    // Link against system libraries needed for LLVM
    for mut arg in llvm_config(&llvm_config_path, "--system-libs")
        .display()
        .to_string()
        .split(&[' ', '\n'])
    {
        if arg.is_empty() {
            continue;
        }
        arg = arg.strip_prefix("-llib").unwrap_or(arg);
        arg = arg.strip_prefix("-l").unwrap_or(arg);
        arg = arg.strip_suffix(".tbd").unwrap_or(arg); // macos
        arg = arg.strip_suffix(".lib").unwrap_or(arg); // msvc
        println!("cargo:rustc-link-lib=dylib={arg}");
    }

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=src/lib.rs");

    let bindings = bindgen::builder()
        .clang_args(
            include_dirs
                .iter()
                .map(|dir| format!("-I{}", dir.as_ref().display())),
        )
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .unwrap();

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();
}
