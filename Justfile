set shell := ["sh", "-c"]
export CARGO_TERM_COLOR := "always"

test: test-std-x86_64 test-std-x86_64-no-std test-std-x86_64-no-std-or-libm

check-thumbv7em: check-thumbv7em-no-std check-thumbv7em-no-std-or-libm

[group('testing')]
test-std-x86_64:
    cargo clippy --features "std" -- -D warnings
    cargo build --no-default-features --features "std"
    cargo test --features "std"

[group('testing')]
test-std-x86_64-no-std:
    cargo clippy --no-default-features --features "libm" -- -D warnings
    cargo build --no-default-features --features "libm"
    cargo test --no-default-features --features "libm"

[group('testing')]
test-std-x86_64-no-std-or-libm:
    cargo clippy --no-default-features -- -D warnings
    cargo build --no-default-features
    cargo test --no-default-features

[group('testing')]
check-thumbv7em-no-std:
    rustup target add thumbv7em-none-eabihf
    cargo clippy --target thumbv7em-none-eabihf --no-default-features --features "libm" -- -D warnings
    cargo build --target thumbv7em-none-eabihf --no-default-features --features "libm"

[group('testing')]
check-thumbv7em-no-std-or-libm:
    rustup target add thumbv7em-none-eabihf
    cargo clippy --target thumbv7em-none-eabihf --no-default-features -- -D warnings
    cargo build --target thumbv7em-none-eabihf --no-default-features

coverage:
    cargo llvm-cov --html --open

# Clean build artifacts
clean:
    cargo clean

# Dry-run a publish (to check for packaging errors without uploading)
check-publish:
    cargo publish --dry-run
