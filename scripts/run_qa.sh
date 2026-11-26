#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[RUN-QA] 1/3 Running unit tests (--test-threads=1)..."
cargo test -- --test-threads=1

echo "[RUN-QA] 2/3 Building release binary..."
cargo build --release

echo "[RUN-QA] 3/3 Running full QA suite..."
./target/release/cubex-rs --qa-suite "$@"
