#!/bin/bash

# Check if 'non-local' is passed as an argument
SAVE_LOCAL_FLAG="-v SAVE_LOCAL=1"
if [ "$1" == "non-local" ]; then
    echo "🚫 Local save disabled: Checkpoints will remain in /dev/shm."
    SAVE_LOCAL_FLAG="-v SAVE_LOCAL=0"
else
    echo "💾 Local save enabled (Default): Will copy checkpoint back to NFS at the end."
fi

# Function to submit and watch a job
run_task() {
    local PBS_SCRIPT=$1
    local LOG_PREFIX=$2
    local EXTRA_ARGS=$3
    
    echo "========================================================"
    echo "🚀 Submitting $PBS_SCRIPT..."
    
    # Pass the SAVE_LOCAL flag if it exists
    QSUB_OUTPUT=$(qsub $EXTRA_ARGS $PBS_SCRIPT)
    if [ $? -ne 0 ]; then
        echo "❌ Submit failed: $QSUB_OUTPUT"
        return 1
    fi
    
    JOB_ID=$(echo "$QSUB_OUTPUT" | sed 's/\..*//' | sed 's/[^0-9]*//g')
    echo "✅ Job ID: $JOB_ID"
    
    # Wait for output files to appear
    echo "⏳ Waiting for job start..."
    STDOUT_FILE="${LOG_PREFIX}.run${4}" # e.g. stdout.run_all
    STDERR_FILE="${LOG_PREFIX/stdout/stderr}.run${4}"
    
    rm -f $STDOUT_FILE $STDERR_FILE
    
    while true; do
        if [ -e "$STDOUT_FILE" ]; then break; fi
        if ! qstat $JOB_ID >/dev/null 2>&1; then 
            echo "⚠️ Job finished/died before creating stdout."
            return 1
        fi
        sleep 2
    done
    
    echo "📺 Streaming Output ($STDOUT_FILE)..."
    tail -f $STDOUT_FILE &
    TAIL_PID=$!
    
    # Wait for finish
    while qstat $JOB_ID >/dev/null 2>&1; do
        sleep 5
    done
    
    kill $TAIL_PID 2>/dev/null
    wait $TAIL_PID 2>/dev/null
    
    echo "✅ Job $JOB_ID finished."
    
    # Print logs
    echo "--- FULL STDOUT ---"
    cat $STDOUT_FILE
    echo "-------------------"
    if [ -s "$STDERR_FILE" ]; then
        echo "--- FULL STDERR ---"
        cat $STDERR_FILE
        echo "-------------------"
    fi
    
    if grep -q "Traceback" $STDERR_FILE; then
        echo "❌ Job seemed to fail (Traceback found)."
        return 1
    fi
    
    return 0
}

# --- Main Flow ---
run_task "run_all.pbs" "stdout" "$SAVE_LOCAL_FLAG" "_all"

if [ $? -ne 0 ]; then
    echo "⛔ Task failed. Aborting."
    exit 1
fi

echo "🎉 All tasks completed successfully!"