import os
import tqdm
import argparse
from h5py import File
from multiprocessing import Process, Queue
from wespeaker import load_model

GPU_NUM = 2

# === 解析命令行参数 ===
parser = argparse.ArgumentParser(
    description="Extract xvectors from audio directory"
)
parser.add_argument(
    "--dir_path",
    type=str,
    required=True,
    help="Path to the directory containing audio files"
)
parser.add_argument(
    "--ext", type=str, default=".mp3", help="Audio file extension to look for"
)
parser.add_argument(
    "--h5_save_path", type=str, required=True, help="Path to save the h5 file"
)

args = parser.parse_args()
dataset_name = args.dir_path.split('/')[-1]
# === 加载模型（避免多进程冲突） ===
_ = load_model('english')  # 触发模型下载

# === 获取参数 ===
dir_path = args.dir_path
ext = args.ext
h5_save_path = args.h5_save_path
intermediate_dir = os.path.dirname(h5_save_path)
os.makedirs(intermediate_dir, exist_ok=True)
# === 获取所有音频文件路径 ===
all_files = []
for dir, _, file_names in os.walk(dir_path):
    for file in file_names:
        if file.endswith(ext):
            all_files.append((dir, file))

print(f'✅ Found {len(all_files)} files in "{dir_path}" with extension "{ext}"')


# 子进程工作函数
def worker(gpu_id, file_list, queue):
    print(f"[GPU:{gpu_id}] 开始")
    from wespeaker import load_model
    import numpy as np
    model = load_model('english')
    model.set_device(f'cuda:{gpu_id}')
    result = {}
    failed = []

    for dir, file in tqdm.tqdm(file_list, desc=f"[GPU:{gpu_id}]"):
        audio_path = os.path.join(dir, file)
        #"l2arctic_ASI_arctic_a0017"
        try:
            emb = model.extract_embedding(audio_path)
            utt_id = file.replace(ext, '')
            if dataset_name == "L2arctic":
                parent_dir_name = dir.split('/')[-2]
                utt_id = f"l2arctic_{parent_dir_name}_{utt_id}"
            print(f"uttid:{utt_id}")
            result[utt_id] = emb
        except Exception as e:
            print(f"❌ [GPU:{gpu_id}] 失败: {audio_path} - {e}")
            failed.append((dir, file))
    print(f"[GPU:{gpu_id}] ✅ 处理完成,开始保存临时文件")
    # 保存本地临时h5
    temp_h5_path = os.path.join(intermediate_dir, f"xvector_{gpu_id}.h5")
    with File(temp_h5_path, 'w') as hf:
        hf.create_group('xvector')
        for utt_id, emb in result.items():
            hf['xvector'][utt_id] = emb
    print(f"[GPU:{gpu_id}] ✅ 临时保存到 {temp_h5_path}")

    # 保存失败日志
    if failed:
        fail_log = os.path.join(intermediate_dir, f'failed_{gpu_id}.txt')
        with open(fail_log, 'w') as f:
            for dir, file in failed:
                f.write(os.path.join(dir, file) + '\n')
        print(f"[GPU:{gpu_id}] ⚠️ 写入失败日志: {fail_log}")

    queue.put((len(result), len(failed)))  # 汇总统计用


# 工具函数：均分列表
def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [
        lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
    ]


# 进程调度
num_gpus = GPU_NUM  # 👈 根据你服务器的显卡数设置
file_chunks = chunk_list(all_files, num_gpus)
queue = Queue()
processes = []

for gpu_id in range(num_gpus):
    p = Process(target=worker, args=(gpu_id, file_chunks[gpu_id], queue))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

# 汇总结果
total_embeds = {}
total_failed = []

for gpu_id in range(num_gpus):
    # 加载中间 H5
    temp_h5_path = os.path.join(intermediate_dir, f"xvector_{gpu_id}.h5")
    with File(temp_h5_path, 'r') as hf:
        for utt_id in hf['xvector']:
            total_embeds[utt_id] = hf['xvector'][utt_id][()]

    # 加载失败日志
    fail_log = os.path.join(intermediate_dir, f'failed_{gpu_id}.txt')
    if os.path.exists(fail_log):
        with open(fail_log, 'r') as f:
            total_failed.extend([line.strip() for line in f.readlines()])

# 保存最终合并版
with File(h5_save_path, 'w') as hf:
    hf.create_group('xvector')
    for utt_id, emb in total_embeds.items():
        hf['xvector'][utt_id] = emb
print(f"\n✅ 汇总保存到 {h5_save_path}")

# 保存失败日志
if total_failed:
    final_fail_log = os.path.join(intermediate_dir, 'failed.txt')
    with open(final_fail_log, 'w') as f:
        for path in total_failed:
            f.write(path + '\n')
    print(f"⚠️ 汇总失败数: {len(total_failed)}，保存至: {final_fail_log}")
else:
    print("✅ 所有音频处理成功，无失败记录。")
