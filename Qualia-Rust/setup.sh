#!/bin/bash
# Setup script for Qualia-Rust
# Author: DarkLightX/Dana Edwards

set -e

echo "🦀 Qualia-Rust Setup Script"
echo "=========================="

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✅ Rust/Cargo detected: $(cargo --version)"

# Add required components
echo "📦 Installing required components..."
rustup component add rustfmt clippy

# Build the project
echo "🔨 Building Qualia-Rust..."
cargo build --release

# Run tests
echo "🧪 Running tests..."
cargo test

# Show help
echo ""
echo "✨ Build complete! You can now use Guardian:"
echo ""
echo "Show help:"
echo "  cargo run --bin guardian -- --help"
echo ""
echo "Analyze a project:"
echo "  cargo run --bin guardian -- analyze /path/to/project --run-quality"
echo ""
echo "For optimized performance, use the release build:"
echo "  ./target/release/guardian --help"
echo ""