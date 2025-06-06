#!/bin/bash
# Fix malformed sensor signatures

cd /Users/danax/projects/QualiaGuardian/Qualia-Rust

# Fix all sensor files
for file in crates/sensors/src/*.rs; do
  echo "Fixing $file"
  sed -i '' 's/_context: context: &SensorContextSensorContext/_context: \&SensorContext/g' "$file"
done

echo "Done!"