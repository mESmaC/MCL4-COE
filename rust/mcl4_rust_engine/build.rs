fn main() {
    println!("cargo:rustc-link-search=native=X:/MCL4-COE/src/mcl4_coe/csrc");
    println!("cargo:rustc-link-lib=dylib=mcl4_deepseek_v41_sm120");
    println!("cargo:rerun-if-changed=X:/MCL4-COE/src/mcl4_coe/csrc/mcl4_deepseek_v41_sm120.cu");
}
