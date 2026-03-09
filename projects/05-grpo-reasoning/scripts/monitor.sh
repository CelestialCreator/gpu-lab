#!/bin/bash
# GRPO Training Monitor with Desktop Notifications
# Usage: ./monitor.sh [check_interval_seconds]
# Sends desktop notification on: completion, crash, reward milestones

INTERVAL=${1:-300}  # default: check every 5 minutes
JOB_NAME="grpo-train"
LAST_REWARD=0
LAST_STEP=0
MILESTONE_REWARDS=(0.4 0.5 0.6 0.7 0.8)
NOTIFIED_MILESTONES=()

notify() {
    local title="$1"
    local msg="$2"
    echo "[$(date '+%H:%M:%S')] $title: $msg"
    # Desktop notification (works over SSH with DISPLAY forwarded)
    notify-send "$title" "$msg" 2>/dev/null || true
    # Also write to log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $title: $msg" >> /home/akshay/grpo-monitor.log
}

check_milestone() {
    local reward=$1
    for m in "${MILESTONE_REWARDS[@]}"; do
        local already=false
        for n in "${NOTIFIED_MILESTONES[@]}"; do
            [[ "$n" == "$m" ]] && already=true
        done
        if ! $already && (( $(echo "$reward >= $m" | bc -l) )); then
            notify "GRPO Milestone" "Reward reached $m! (current: $reward)"
            NOTIFIED_MILESTONES+=("$m")
        fi
    done
}

notify "GRPO Monitor" "Started monitoring job/$JOB_NAME (checking every ${INTERVAL}s)"

while true; do
    # Check pod status
    STATUS=$(kubectl get pods -l job-name=$JOB_NAME -o jsonpath='{.items[0].status.phase}' 2>/dev/null)

    if [[ "$STATUS" == "Succeeded" ]]; then
        notify "GRPO COMPLETE" "Training finished successfully! Run eval now."
        break
    elif [[ "$STATUS" == "Failed" ]] || [[ "$STATUS" == "" ]]; then
        notify "GRPO ALERT" "Job status: ${STATUS:-NOT FOUND}. Check kubectl logs."
        break
    fi

    # Parse latest metrics
    LATEST=$(kubectl logs job/$JOB_NAME --tail=3 2>/dev/null | grep "reward" | tail -1)
    if [[ -n "$LATEST" ]]; then
        REWARD=$(echo "$LATEST" | grep -oP "'reward': '?[\d.]+" | grep -oP "[\d.]+$" | tail -1)
        MATH_R=$(echo "$LATEST" | grep -oP "'rewards/math_reward/mean': '?[\d.]+" | grep -oP "[\d.]+$" | tail -1)
        STEP=$(echo "$LATEST" | grep -oP "'epoch': '?[\d.]+" | grep -oP "[\d.]+$" | tail -1)
        STEP_TIME=$(echo "$LATEST" | grep -oP "'step_time': '?[\d.]+" | grep -oP "[\d.]+$" | tail -1)

        if [[ -n "$REWARD" ]]; then
            # Calculate approximate progress
            PROGRESS=$(echo "$STEP * 100 / 3" | bc -l 2>/dev/null | head -c5)
            echo "[$(date '+%H:%M:%S')] reward=$REWARD math=$MATH_R epoch=$STEP step_time=${STEP_TIME}s progress~${PROGRESS}%"
            check_milestone "$REWARD"
        fi
    fi

    # Check GPU health
    GPU_UTIL=$(kubectl exec job/$JOB_NAME -- nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %')
    if [[ -n "$GPU_UTIL" ]] && (( GPU_UTIL < 5 )); then
        notify "GRPO WARNING" "GPU utilization is ${GPU_UTIL}% - possible hang"
    fi

    sleep $INTERVAL
done
