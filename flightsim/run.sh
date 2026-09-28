#!/usr/bin/env bash
# Launch the flight sim screensaver fullscreen. Extra args go to the hack
# (e.g. --time dusk --no-hud --speed 1.5 --fov 70 --seed 42).
# Built from flightsim.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/flightsim" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Flightsim"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Flightsim"] fullscreen enable'

fi
wait "$pid"
