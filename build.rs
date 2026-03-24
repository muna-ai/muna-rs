/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

#[cfg(unix)]
use std::os::unix::fs as unix_fs;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let version = fxnc_version();
    let version_marker = out_dir.join(".fxnc-version");
    let cached_version = fs::read_to_string(&version_marker).unwrap_or_default();
    let needs_download = cached_version.trim() != version;
    let downloads = cdn_downloads(&target_os, &target_arch);
    for (remote_name, local_name) in &downloads {
        let dest = out_dir.join(local_name);
        if needs_download || !dest.exists() {
            let url = format!("https://cdn.fxn.ai/fxnc/{version}/{remote_name}");
            download(&url, &dest);
        }
    }
    if needs_download {
        fs::write(&version_marker, &version).expect("Failed to write version marker");
    }
    if target_os == "macos" {
        let symlink = out_dir.join("Function.dylib");
        let _ = fs::remove_file(&symlink);
        #[cfg(unix)]
        unix_fs::symlink("libFunction.dylib", &symlink)
            .expect("Failed to create Function.dylib symlink");
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=Function");
    if target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
    println!("cargo:lib_dir={}", out_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
}

fn cdn_downloads(os: &str, arch: &str) -> Vec<(&'static str, &'static str)> {
    match (os, arch) {
        ("macos", "aarch64") => vec![
            ("Function-macos-arm64.dylib", "libFunction.dylib"),
        ],
        ("windows", "x86_64") => vec![
            ("Function-win-x86_64.dll", "Function.dll"),
            ("Function-win-x86_64.lib", "Function.lib"),
        ],
        ("windows", "aarch64") => vec![
            ("Function-win-arm64.dll", "Function.dll"),
            ("Function-win-arm64.lib", "Function.lib"),
        ],
        ("linux", "x86_64") => vec![
            ("libFunction-linux-x86_64.so", "libFunction.so"),
        ],
        ("linux", "aarch64") => vec![
            ("libFunction-linux-arm64.so", "libFunction.so"),
        ],
        _ => panic!("Unsupported platform: {os}-{arch}"),
    }
}

fn fxnc_version() -> String {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cargo_toml = fs::read_to_string(PathBuf::from(manifest_dir).join("Cargo.toml"))
        .expect("Failed to read Cargo.toml");
    let mut in_fxnc_section = false;
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_fxnc_section = trimmed == "[package.metadata.fxnc]";
            continue;
        }
        if in_fxnc_section && trimmed.starts_with("version") {
            if let Some(start) = trimmed.find('"') {
                if let Some(end) = trimmed[start + 1..].find('"') {
                    return trimmed[start + 1..start + 1 + end].to_string();
                }
            }
        }
    }
    panic!("Failed to find fxnc version in Cargo.toml [package.metadata.fxnc]");
}

fn download(url: &str, dest: &PathBuf) {
    eprintln!("Downloading {url}");
    let response = ureq::get(url).call().expect(&format!("Failed to download {url}"));
    let mut bytes = Vec::new();
    response
        .into_body()
        .as_reader()
        .read_to_end(&mut bytes)
        .expect(&format!("Failed to read response from {url}"));
    fs::write(dest, &bytes).expect(&format!("Failed to write {}", dest.display()));
}
