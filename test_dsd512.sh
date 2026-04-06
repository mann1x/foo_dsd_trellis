#!/bin/bash
# DSD512 start/stop stress test — 50 cycles, 12s play each
LOG="C:/Users/manni/AppData/Roaming/foobar2000-v2/user-components-x64/foo_dsd_trellis/foo_dsd_trellis.log"
baseline=$(grep -c "underrun\|BUG\|error\|crash\|rate not\|terminat" "$LOG" 2>/dev/null)
echo "Starting 50-cycle DSD512 test (baseline errors=$baseline)"
for i in $(seq 1 50); do
    curl -s -X POST http://localhost:8880/api/player/play/8/3 > /dev/null 2>&1
    sleep 12
    curl -s -X POST http://localhost:8880/api/player/stop > /dev/null 2>&1
    sleep 3
    now=$(grep -c "underrun\|BUG\|error\|crash\|rate not\|terminat" "$LOG" 2>/dev/null)
    new=$((now - baseline))
    echo "cycle $i/50: new_errors=$new"
    if [ "$new" -gt 0 ]; then
        echo "  ERROR DETECTED — last 5 error lines:"
        grep "underrun\|BUG\|error\|crash\|rate not\|terminat" "$LOG" | tail -5
    fi
done
echo "DONE: total new errors=$((now - baseline))"
