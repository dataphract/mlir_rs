// Derived from Fabian Schuiki's build script for `moore`.

use std::{env, path::PathBuf};

use build_common::{llvm_config, setup_llvm_build};

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let base_dir = crate_dir.parent().unwrap().parent().unwrap();
    let circt_dir = base_dir.join("circt");
    let llvm_dir = circt_dir.join("llvm");
    let llvm_build_dir = llvm_dir.join("build");

    let llvm_config_path = llvm_build_dir.join("bin").join("llvm-config");
    let llvm_include_dir = llvm_config(&llvm_config_path, "--includedir");

    let lib_names = [
        "LLVMBinaryFormat",
        "LLVMBitstreamReader",
        "LLVMCore",
        "LLVMRemarks",
        "LLVMSupport",
        "MLIRAnalysis",
        "MLIRArithDialect",
        "MLIRAsmParser",
        "MLIRBytecodeReader",
        "MLIRCAPIFunc",
        "MLIRCAPIIR",
        "MLIRCAPIControlFlow",
        "MLIRCallInterfaces",
        "MLIRControlFlowDialect",
        "MLIRControlFlowInterfaces",
        "MLIRFuncDialect",
        "MLIRIR",
        "MLIRInferTypeOpInterface",
        "MLIRInferIntRangeInterface",
        "MLIRPDLDialect",
        "MLIRPDLInterpDialect",
        "MLIRPDLToPDLInterp",
        "MLIRParser",
        "MLIRPass",
        "MLIRRewrite",
        "MLIRSideEffectInterfaces",
        "MLIRSupport",
        "MLIRTransformUtils",
        "MLIRTransforms",
    ];

    // Make a list of include directories.
    let include_dirs = vec![
        llvm_include_dir,
        llvm_dir.join("llvm/include"),
        llvm_dir.join("mlir/include"),
        llvm_build_dir.join("include"),
        llvm_build_dir.join("tools/mlir/include"),
    ];

    setup_llvm_build(&lib_names, &include_dirs);
}
