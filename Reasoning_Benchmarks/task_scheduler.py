import os
import time
import subprocess
import logging
from collections import deque
from pynvml import *

# ==============================================================================
# 1. 配置区域 - 请根据您的环境修改此部分
# ==============================================================================

# 监控的GPU配对
GPU_PAIRS = [(0, 1), (2, 3), (4, 5),(6,7)]

# 【新】包含待执行任务列表的文本文件名
# 您可以在脚本运行时随时向此文件添加新的任务目录名
TASK_FILE = "tasks_to_run.txt"

# 基础工作目录
BASE_WORKDIR = "automated_runs"

# --- 阈值配置 ---

# 判定为"空闲"需要达到的连续时间（分钟）
IDLE_THRESHOLD_MINUTES = 3

# 监控检查的时间间隔（秒）
MONITORING_INTERVAL_SECONDS = 60

# 每隔多少个监控周期（上面的秒数）就重新读取一次任务文件
TASK_CHECK_INTERVAL_CYCLES = 5 # 5 * 60s = 5分钟

# GPU被视为"空闲"的显存占用阈值（GB）
IDLE_MEM_THRESHOLD_GB = 4

# ==============================================================================
# 2. 脚本核心逻辑 - 通常不需要修改
# ==============================================================================

# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler("scheduler.log")])

IDLE_MEM_THRESHOLD_BYTES = IDLE_MEM_THRESHOLD_GB * 1024**3

class GPUTaskScheduler:
    def __init__(self):
        # 任务状态跟踪
        self.pending_tasks = deque()
        self.running_tasks = set()
        self.completed_tasks = set()
        
        self.idle_start_time = {pair: None for pair in GPU_PAIRS}
        # 现在存储更详细的进程信息
        self.running_processes = {pair: None for pair in GPU_PAIRS}

        try:
            nvmlInit()
            self.gpu_handles = {i: nvmlDeviceGetHandleByIndex(i) for i in range(nvmlDeviceGetCount())}
            logging.info(f"成功初始化NVML，检测到 {len(self.gpu_handles)} 块GPU。")
        except NVMLError as error:
            logging.error(f"初始化NVML失败: {error}")
            raise

        # 首次加载任务
        self._update_tasks_from_file()

    # ========================================================================
    # v v v v v v v v v v v v  唯一的修改在这里 v v v v v v v v v v v v
    #
    def _update_tasks_from_file(self):
        """
        【新逻辑】从任务文件重新构建待处理任务队列。
        这会同步文件中的添加、删除和顺序变更。
        """
        logging.info(f"正在从 {TASK_FILE} 同步任务列表...")
        try:
            if not os.path.exists(TASK_FILE):
                logging.warning(f"任务文件 {TASK_FILE} 不存在，清空待处理队列。")
                self.pending_tasks.clear()
                return

            with open(TASK_FILE, 'r') as f:
                # 读取文件中的所有有效任务，保持其顺序
                tasks_in_file = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            # 创建一个新的待处理队列
            new_pending_tasks = deque()
            for task in tasks_in_file:
                # 只有当一个任务既没有在运行，也没有完成时，才应被视为“待处理”
                if task not in self.running_tasks and task not in self.completed_tasks:
                    new_pending_tasks.append(task)
            
            # 用新队列替换旧队列
            if self.pending_tasks != new_pending_tasks:
                logging.info(f"任务队列已更新。旧队列大小: {len(self.pending_tasks)}, 新队列大小: {len(new_pending_tasks)}")
                self.pending_tasks = new_pending_tasks
            else:
                logging.info("任务队列与文件内容一致，无变化。")

        except Exception as e:
            logging.error(f"读取或同步任务文件时出错: {e}")

    #
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^  唯一的修改在这里 ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
    # ========================================================================

    def is_gpu_idle(self, gpu_index):
        """根据显存占用绝对值判断GPU是否空闲"""
        try:
            handle = self.gpu_handles[gpu_index]
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            is_idle = memory_info.used < IDLE_MEM_THRESHOLD_BYTES
            mem_used_gb = memory_info.used / (1024**3)
            logging.debug(f"GPU {gpu_index}: MemUsed={mem_used_gb:.2f}GB -> IsIdle={is_idle}")
            return is_idle
        except NVMLError as error:
            logging.error(f"无法获取GPU {gpu_index} 的状态: {error}")
            return False

    def launch_task(self, gpu_pair):
        """为指定的GPU配对启动一个任务"""
        if not self.pending_tasks:
            logging.warning("任务队列已空空，无法启动新任务。")
            return

        task_to_run = self.pending_tasks.popleft()
        self.running_tasks.add(task_to_run)
        
        gpu_ids_str = ",".join(map(str, gpu_pair))
        task_full_path = os.path.join(BASE_WORKDIR, task_to_run)
        original_script_path = os.path.join(task_full_path, "run_eval_all.sh")

        if not os.path.exists(original_script_path):
            logging.error(f"在 {task_full_path} 中未找到 run_eval_all.sh，任务 '{task_to_run}' 已跳过。")
            self.running_tasks.remove(task_to_run)
            self.completed_tasks.add(task_to_run) # 标记为完成（虽然是失败的）以防重试
            return

        # ... 文件读写和修改部分，逻辑不变 ...
        with open(original_script_path, 'r') as f: script_content = f.read()
        modified_content = []
        found = False
        for line in script_content.splitlines():
            if line.strip().startswith('export CUDA_VISIBLE_DEVICES='):
                modified_content.append(f'export CUDA_VISIBLE_DEVICES="{gpu_ids_str}"')
                found = True
            else: modified_content.append(line)
        if not found:
            modified_content.insert(1, f'export CUDA_VISIBLE_DEVICES="{gpu_ids_str}"')
        modified_script_content = "\n".join(modified_content)
        temp_script_name = f"run_temp_{gpu_ids_str.replace(',', '_')}.sh"
        temp_script_path = os.path.join(task_full_path, temp_script_name)
        with open(temp_script_path, 'w') as f: f.write(modified_script_content)
        os.chmod(temp_script_path, 0o755)

        log_file_path = os.path.join(task_full_path, f"run_log_{gpu_ids_str.replace(',', '_')}.log")
        command = ["/bin/bash", temp_script_name]

        logging.info(f"在GPU {gpu_ids_str} 上启动任务: {task_to_run}")
        logging.info(f"执行命令: {' '.join(command)}，工作目录: {task_full_path}")
        logging.info(f"日志将输出到: {log_file_path}")

        try:
            with open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(command, cwd=task_full_path, stdout=log_file, stderr=subprocess.STDOUT)
            self.running_processes[gpu_pair] = {'process': process, 'task': task_to_run}
        except Exception as e:
            logging.error(f"启动任务 {task_to_run} 失败: {e}")
            self.running_tasks.remove(task_to_run)
            self.pending_tasks.appendleft(task_to_run) # 启动失败，放回队列头部重试

    def run(self):
        """主监控循环"""
        task_check_counter = 0
        while True:
            # 1. 检查已完成的进程
            for pair, process_info in list(self.running_processes.items()):
                if process_info and process_info['process'].poll() is not None:
                    task = process_info['task']
                    process = process_info['process']
                    if process.returncode == 0:
                        logging.info(f"任务 '{task}' 在 GPU {pair} 上已成功完成。")
                    else:
                        logging.warning(f"任务 '{task}' 在 GPU {pair} 上已结束，返回码: {process.returncode}。请检查日志。")
                    
                    # 更新任务状态
                    self.running_tasks.remove(task)
                    self.completed_tasks.add(task)
                    logging.info(f"已完成任务数: {len(self.completed_tasks)}")
                    
                    # 释放GPU资源记录
                    self.running_processes[pair] = None
                    self.idle_start_time[pair] = None

            # 2. 检查空闲GPU并启动新任务
            for pair in GPU_PAIRS:
                if self.running_processes.get(pair):
                    logging.debug(f"GPU {pair} 正在运行任务，跳过空闲检查。")
                    continue
                
                gpu1, gpu2 = pair
                if self.is_gpu_idle(gpu1) and self.is_gpu_idle(gpu2):
                    if self.idle_start_time[pair] is None:
                        self.idle_start_time[pair] = time.time()
                        logging.info(f"GPU {pair} 开始进入空闲状态。")
                    
                    idle_duration = (time.time() - self.idle_start_time[pair]) / 60
                    logging.info(f"GPU {pair} 已连续空闲 {idle_duration:.2f} 分钟。")

                    if idle_duration >= IDLE_THRESHOLD_MINUTES:
                        if self.pending_tasks:
                            self.launch_task(pair)
                            self.idle_start_time[pair] = None # 重置计时器
                        elif not self.running_tasks: # 队列空了，也没有在跑的任务
                            logging.info("所有已知任务已完成，等待新任务加入文件...")

                else: # GPU不再空闲
                    if self.idle_start_time[pair] is not None:
                         logging.info(f"GPU {pair} 不再处于空闲状态，重置计时器。")
                    self.idle_start_time[pair] = None

            # 3. 周期性检查任务文件
            task_check_counter += 1
            if task_check_counter >= TASK_CHECK_INTERVAL_CYCLES:
                self._update_tasks_from_file()
                task_check_counter = 0

            time.sleep(MONITORING_INTERVAL_SECONDS)

    def shutdown(self):
        """清理资源"""
        try:
            nvmlShutdown()
            logging.info("NVML已成功关闭。")
        except NVMLError as error:
            logging.error(f"关闭NVML时出错: {error}")

if __name__ == "__main__":
    scheduler = GPUTaskScheduler()
    try:
        scheduler.run()
    except KeyboardInterrupt:
        logging.info("接收到中断信号，正在关闭调度器...")
    finally:
        scheduler.shutdown()