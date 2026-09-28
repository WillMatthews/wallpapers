#!/usr/bin/env bash
# Print the next lock screensaver from a shuffle bag: every saver once per
# cycle in random order, and a new cycle never opens with the one just shown
# (plain shuf streaked badly). Logs each pick to lock-saver.log.
STATE=${XDG_STATE_HOME:-$HOME/.local/state}
BAG=$STATE/lock-saver-bag
read -r -a bag 2>/dev/null < "$BAG" || bag=()
if ((${#bag[@]} == 0)); then
    last=$(tail -n1 "$STATE/lock-saver.log" 2>/dev/null | awk '{print $NF}')
    mapfile -t bag < <(shuf -e pipes maze flightsim)
    [[ ${bag[0]} == "$last" ]] && bag=("${bag[@]:1}" "${bag[0]}")
fi
echo "${bag[@]:1}" > "$BAG"
echo "$(date -Is) ${bag[0]}" >> "$STATE/lock-saver.log"
echo "${bag[0]}"
