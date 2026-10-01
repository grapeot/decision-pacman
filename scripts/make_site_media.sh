#!/usr/bin/env bash
# Builds the site's web clips from docs/media (originals stay untouched): the browser clips are
# cropped to the maze, the iPhone clip is scaled down; all H.264 with faststart, plus poster frames.
set -euo pipefail
cd "$(dirname "$0")/.."
out=site/public/media
mkdir -p "$out"
for name in jev phi4_mini pacman_08b_qwen; do
  ffmpeg -v error -y -i "docs/media/demo_${name}.mp4" -vf "crop=558:618:43:17" \
    -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -profile:v high -movflags +faststart -an "$out/${name}.mp4"
  ffmpeg -v error -y -ss 1.5 -i "$out/${name}.mp4" -frames:v 1 -q:v 5 "$out/${name}.jpg"
done
ffmpeg -v error -y -i docs/media/iphone_pacman_08b_qwen.mp4 -vf "scale=440:-2" \
  -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 64k "$out/iphone_08b.mp4"
ffmpeg -v error -y -ss 3 -i "$out/iphone_08b.mp4" -frames:v 1 -q:v 5 "$out/iphone_08b.jpg"
ls -la "$out"
