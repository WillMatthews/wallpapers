#!/usr/bin/env bash
# Lock-screen background for swaylock-plugin --command-each (one instance per
# output). Runs a screensaver on the root window of a private Xwayland, which
# windowtolayer turns into a layer surface behind the password ring.
# Usage: lock-bg.sh [pipes|maze|flightsim] [seed] [epoch]   (default: random)
# With no args it reads "saver seed epoch" from $XDG_RUNTIME_DIR/lock-bg.cur,
# which lock.sh and lock-next.sh write; swaylock-plugin re-runs this same
# command whenever the saver dies, so that file is how a skip takes effect.
# flightsim spans all outputs as one scene: every instance needs the same seed
# and epoch, plus the sway layout and which output it's on.
dir="$(cd "$(dirname "$0")" && pwd)"
cur=${XDG_RUNTIME_DIR:-/tmp}/lock-bg.cur
(($# == 0)) && [[ -r $cur ]] && read -r -a args < "$cur" && set -- "${args[@]}"
pick=${1:-$(shuf -n1 -e pipes maze flightsim)}
case $pick in
  pipes)     cmd=("$dir/pipes/bin/pipes" --root --delay 5000) ;;
  maze)      cmd=("$dir/maze3d/bin/maze3d" --root) ;;
  flightsim)
    cmd=("$dir/flightsim/flightsim" --root)
    [[ -n ${2:-} ]] && cmd+=(--seed "$2")
    [[ -n ${3:-} ]] && cmd+=(--epoch "$3")
    # "x,y,w,h;..." of the active outputs, and our index in it
    if [[ -n ${SWAYLOCK_PLUGIN_OUTPUT_NAME:-} ]] &&
       read -r layout idx < <(swaymsg -t get_outputs -r 2>/dev/null | jq -r --arg me "$SWAYLOCK_PLUGIN_OUTPUT_NAME" '
         [.[] | select(.active)] as $o
         | ([$o[].rect | "\(.x),\(.y),\(.width),\(.height)"] | join(";")) + " "
         + ([$o[].name] | index($me) // -1 | tostring)') && ((idx >= 0)); then
      cmd+=(--layout "$layout" --output "$idx")
    fi ;;
  *) echo "lock-bg.sh: unknown screensaver '$pick'" >&2; exit 1 ;;
esac
exec "$dir/build/windowtolayer/target/release/windowtolayer" \
  "$dir/build/swaylock-plugin/example_xwayland_wrapper.py" "${cmd[@]}"
