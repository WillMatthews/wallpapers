#!/usr/bin/env bash
# Launch the Pipes screensaver fullscreen. Extra args go to the hack
# (e.g. --count 3 --size 400 --fisheye --tightturns --factory 2 --fps).
# Binary is xscreensaver 6.08's "pipes" (Ubuntu xscreensaver-gl), vendored in bin/.
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
# --delay is µs between frames (hack default 10000); lower = faster pipes
"$dir/bin/pipes" --window --delay 5000 "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Pipes"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Pipes"] fullscreen enable'

fi
wait "$pid"
