#!/bin/bash

# Configuration
NUM_RUNS=100  # Number of times to run each configuration
DELTA_K_VALUES=(20)  # Different cautious_delta_k values to test
BRAKE_THRESHOLD_VALUES=(0.3 0.5 0.7)  # Different emergency_brake_threshold values to test
CONFIG_TYPES=("right_turn")
RESULTS_DIR="experiment_results"
LOG_DIR="experiment_logs"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Function to restart CARLA
restart_carla() {
    echo "Restarting CARLA..."
    docker stop carla >/dev/null 2>&1
    docker rm carla >/dev/null 2>&1
    sleep 2
    docker run -d \
        --name=carla \
        --privileged \
        --gpus all \
        --net=host \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        carlasim/carla:0.9.15 \
        /bin/bash ./CarlaUE4.sh -RenderOffScreen
    sleep 10
}

# Function to run a single experiment
run_experiment() {
    local delta_k=$1
    local config_type=$2
    local brake_threshold=$3
    local run_number=$4
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/experiment_${config_type}_${delta_k}_${brake_threshold}_${run_number}_${timestamp}.log"
    
    echo "Running experiment with cautious_delta_k=$delta_k, config_type=$config_type, emergency_brake_threshold=$brake_threshold (Run $run_number)"
    
    # Run the Python script with the specified parameters
    python3 main_static.py --cautious_delta_k $delta_k --config_type $config_type --emergency_brake_threshold $brake_threshold > "$log_file" 2>&1
    
    # Check if the script crashed
    if [ $? -ne 0 ]; then
        echo "Experiment failed. Restarting CARLA..."
        restart_carla
        return 1
    fi
    
    return 0
}

# Main experiment loop
for config_type in "${CONFIG_TYPES[@]}"; do
    for delta_k in "${DELTA_K_VALUES[@]}"; do
        for brake_threshold in "${BRAKE_THRESHOLD_VALUES[@]}"; do
            echo "Starting experiments with config_type=$config_type, cautious_delta_k=$delta_k, emergency_brake_threshold=$brake_threshold"
            
            for run in $(seq 1 $NUM_RUNS); do
                # Ensure CARLA is running
                # if ! docker ps | grep -q carla; then
                #     restart_carla
                # fi
                
                # Run experiment with retry logic
                max_retries=3
                retry_count=0
                success=false
                
                while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
                    if run_experiment $delta_k $config_type $brake_threshold $run; then
                        success=true
                    else
                        retry_count=$((retry_count + 1))
                        echo "Retry $retry_count of $max_retries"
                        sleep 5
                    fi
                done
                
                if [ "$success" = false ]; then
                    echo "Failed to complete experiment after $max_retries retries"
                fi
                
                # Brief pause between runs
                sleep 5
            done
        done
    done
done

echo "All experiments completed"
