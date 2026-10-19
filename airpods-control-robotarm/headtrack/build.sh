#!/bin/sh
# Build the CoreMotion head-pose streamer.
# The Info.plist must be embedded in the binary (NSMotionUsageDescription) or
# macOS never prompts for motion access and no samples are ever delivered.
set -e
cd "$(dirname "$0")"
swiftc -O main.swift -o headtrack -framework CoreMotion \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist
codesign -s - --force headtrack
echo "built ./headtrack"
