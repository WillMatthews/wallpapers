#!/usr/bin/env bash
# Launch the 3D Maze screensaver fullscreen. Extra args go to the hack
# (e.g. --rats 3 --inverters 5 --rows 20 --columns 20 --speed 2 --drop-acid).
# Binary is xscreensaver 6.08's "maze3d" (Ubuntu xscreensaver-gl), vendored in bin/.
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/bin/maze3d" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Maze3D"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Maze3D"] fullscreen enable'

fi
wait "$pid"
