#!/usr/bin/env bash
# Skip to the next lock screensaver (sway: bindsym --locked $mod+n). Writes the
# new pick for lock-bg.sh, then kills the running savers; swaylock-plugin
# restarts its command on every output and they pick up the new one.
pgrep -x swaylock-plugin >/dev/null || exit 0
dir="$(cd "$(dirname "$0")" && pwd)"
echo "$("$dir/pick-saver.sh") $((SRANDOM | 1)) $(date +%s.%N)" > "${XDG_RUNTIME_DIR:-/tmp}/lock-bg.cur"
# anchored on the full path so this script's own command line can't match
pkill -f "^$dir/(pipes/bin/pipes|maze3d/bin/maze3d|flightsim/flightsim) --(root|layer)"
