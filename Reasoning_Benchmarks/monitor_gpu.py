import subprocess
import time
import sys
from datetime import datetime

# --- Configuration ---
GPU_ID = 0  # Index of the GPU to monitor (usually 0)
MEMORY_THRESHOLD_GIB = 10  # Memory usage threshold in GiB
CONSECUTIVE_MINUTES_TRIGGER = 5  # Number of consecutive minutes below the threshold to trigger the action
CHECK_INTERVAL_SECONDS = 60  # Interval between checks (60 seconds = 1 minute)
SCRIPT_TO_RUN = "bash ./one_eval.sh"  # The script to run when conditions are met

# --- Main Logic ---

def get_gpu_memory_mib(gpu_id: int) -> int | None:
    """
    Fetches the used GPU memory in MiB for a specific GPU using nvidia-smi.
    Returns None if the query fails.
    """
    try:
        # Command to query used memory for a specific GPU in CSV format, without header or units.
        # --query-gpu=memory.used: Queries the used GPU memory.
        # --format=csv,noheader,nounits: Formats the output as CSV, omitting the header and units (e.g., 'MiB').
        # -i {gpu_id}: Specifies the GPU ID.
        command = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}"
        
        # Execute the command and capture the output.
        result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.PIPE)
        
        # Convert the string output to an integer.
        memory_used_mib = int(result.strip())
        return memory_used_mib
    except subprocess.CalledProcessError as e:
        print(f" [ERROR] Failed to execute nvidia-smi: {e.stderr.strip()}", file=sys.stderr)
        return None
    except ValueError:
        print(f" [ERROR] Could not parse the output of nvidia-smi as a number.", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(" [ERROR] nvidia-smi command not found. Please ensure NVIDIA drivers are installed and in your system's PATH.", file=sys.stderr)
        return None


def main():
    """
    Main monitoring loop.
    """
    # Convert the GiB threshold to MiB for comparison, as nvidia-smi outputs in MiB (1 GiB = 1024 MiB).
    memory_threshold_mib = MEMORY_THRESHOLD_GIB * 1024
    
    consecutive_low_usage_count = 0
    
    print("--- GPU Memory Monitor Started ---")
    print(f"Monitoring Target: GPU {GPU_ID}")
    print(f"Memory Threshold: Below {MEMORY_THRESHOLD_GIB} GiB ({memory_threshold_mib} MiB)")
    print(f"Trigger Condition: {CONSECUTIVE_MINUTES_TRIGGER} consecutive minutes meeting the condition")
    print(f"Script to Execute: {SCRIPT_TO_RUN}")
    print("------------------------------------")

    try:
        while True:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            memory_used = get_gpu_memory_mib(GPU_ID)
            
            if memory_used is not None:
                print(f"[{current_time}] Current GPU {GPU_ID} memory usage: {memory_used} MiB")
                
                # Check if memory usage is below the threshold
                if memory_used < memory_threshold_mib:
                    consecutive_low_usage_count += 1
                    print(f"    -> Memory usage is below the threshold. Consecutive count: {consecutive_low_usage_count}/{CONSECUTIVE_MINUTES_TRIGGER}")
                else:
                    print(f"    -> Memory usage is at or above the threshold. Resetting counter.")
                    consecutive_low_usage_count = 0  # Reset the counter
                
                # Check if the trigger condition is met
                if consecutive_low_usage_count >= CONSECUTIVE_MINUTES_TRIGGER:
                    print(f"\n[SUCCESS] GPU {GPU_ID} memory has been below {MEMORY_THRESHOLD_GIB} GiB for {CONSECUTIVE_MINUTES_TRIGGER} consecutive minutes.")
                    print(f"Executing script: {SCRIPT_TO_RUN}")
                    try:
                        # Execute the specified bash script
                        subprocess.run(SCRIPT_TO_RUN, shell=True, check=True)
                        print("[COMPLETE] Script executed successfully. Exiting monitor.")
                        break  # Exit the loop
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] Failed to execute {SCRIPT_TO_RUN}. Exit code: {e.returncode}", file=sys.stderr)
                        break
                    except FileNotFoundError:
                        print(f"[ERROR] Script file {SCRIPT_TO_RUN} not found.", file=sys.stderr)
                        break
            else:
                # Log a message if fetching memory info failed
                print(f"[{current_time}] Failed to get GPU {GPU_ID} memory. Retrying in {CHECK_INTERVAL_SECONDS} seconds.")

            # Wait for the next check
            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n[INFO] Script interrupted by user. Exiting.")


if __name__ == "__main__":
    main()