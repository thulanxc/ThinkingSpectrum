import subprocess
import time
import sys

# --- 可配置参数 ---

# 要监控的GPU设备ID (从0开始)
GPU_ID = 3

# 显存占用的阈值 (单位: GB)
MEMORY_THRESHOLD_GB = 10

# 需要连续低于阈值的分钟数
CONSECUTIVE_MINUTES = 3

# 每次检测的间隔时间 (单位: 秒)
CHECK_INTERVAL_SECONDS = 60

# --- 脚本主逻辑 ---

def get_gpu_memory_usage(gpu_id: int) -> float:
    """
    获取指定ID的GPU的当前显存使用量 (单位: GB)。
    如果出错或找不到GPU，则返回-1。
    """
    try:
        # 运行 nvidia-smi 命令查询指定GPU的已用显存（单位 MiB）
        result = subprocess.run(
            [
                'nvidia-smi',
                f'--id={gpu_id}',
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            check=True
        )
        # 将输出的 MiB 转换为 GB
        used_memory_mib = float(result.stdout.strip())
        used_memory_gb = used_memory_mib / 1024
        return used_memory_gb
    except FileNotFoundError:
        print("错误: `nvidia-smi` 命令未找到。请确保NVIDIA驱动已正确安装并且 `nvidia-smi` 在系统的PATH中。", file=sys.stderr)
        return -1
    except subprocess.CalledProcessError:
        print(f"错误: 无法获取GPU ID {gpu_id} 的信息。请确认该GPU是否存在。", file=sys.stderr)
        return -1
    except Exception as e:
        print(f"发生未知错误: {e}", file=sys.stderr)
        return -1

def run_evaluation_script():
    """
    执行 'bash run_eval_all.sh' 命令。
    """
    print("\n" + "="*50)
    print("条件满足，开始执行 'bash run_eval_all.sh'...")
    print("="*50)
    try:
        # 使用 subprocess.run 执行脚本，并等待其完成
        process = subprocess.run(
            ["bash", "run_eval_all.sh"],
            check=True,
            capture_output=True,
            text=True
        )
        print("脚本执行成功！")
        print("\n--- STDOUT ---")
        print(process.stdout)
        print("\n--- STDERR ---")
        print(process.stderr)
    except FileNotFoundError:
        print("错误: 'run_eval_all.sh' 脚本未在当前目录找到。", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败，返回码: {e.returncode}", file=sys.stderr)
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
    except Exception as e:
        print(f"执行脚本时发生未知错误: {e}", file=sys.stderr)


if __name__ == "__main__":
    consecutive_low_memory_count = 0
    print(f"开始监控 {GPU_ID}号GPU...")
    print(f"监控条件: 显存连续 {CONSECUTIVE_MINUTES} 分钟低于 {MEMORY_THRESHOLD_GB} GB")
    print(f"检测间隔: {CHECK_INTERVAL_SECONDS} 秒")
    print("="*50)

    try:
        while True:
            # 获取当前显存占用
            memory_used = get_gpu_memory_usage(GPU_ID)
            
            # 如果获取失败，则退出脚本
            if memory_used == -1:
                sys.exit(1)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] GPU {GPU_ID} 显存占用: {memory_used:.2f} GB")

            # 检查显存是否低于阈值
            if memory_used < MEMORY_THRESHOLD_GB:
                consecutive_low_memory_count += 1
                print(f"  -> 低于阈值。当前连续计数: {consecutive_low_memory_count}/{CONSECUTIVE_MINUTES}")
            else:
                if consecutive_low_memory_count > 0:
                    print("  -> 高于或等于阈值，重置计数器。")
                consecutive_low_memory_count = 0
            
            # 检查是否已满足连续分钟数的要求
            if consecutive_low_memory_count >= CONSECUTIVE_MINUTES:
                run_evaluation_script()
                break  # 任务完成，退出循环

            # 等待下一个检测周期
            time.sleep(CHECK_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\n用户手动中断脚本。")
        sys.exit(0)