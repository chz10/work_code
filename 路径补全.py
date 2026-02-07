import os
import re
from collections import defaultdict


def extract_timestamps_from_txt(txt_path):
    """从 txt 中读取并提取所有 14 位时间戳"""
    timestamps = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            timestamps.extend(re.findall(r'\d{14}', line))
    return timestamps


def find_videos_by_timestamps(root_dir, timestamps):
    """
    在 root_dir 下递归查找：
    - 优先 h264
    - 若某时间戳只有 h265，也保留
    """
    ts_to_files = defaultdict(list)

    for root, _, files in os.walk(root_dir):
        for f in files:
            lower_f = f.lower()
            if not lower_f.endswith(('.h264', '.h265')):
                continue

            for ts in timestamps:
                if ts in f:
                    full_path = os.path.join(root, f)
                    ts_to_files[ts].append(full_path)
                    break

    return ts_to_files


def select_preferred_videos(ts_to_files):
    """
    对每个时间戳：
    - 优先选择 h264
    - 否则使用 h265
    - 多个结果视为重复
    """
    selected = []
    duplicates = []
    failed = []

    for ts, files in ts_to_files.items():
        h264 = [f for f in files if f.lower().endswith('.h264')]
        h265 = [f for f in files if f.lower().endswith('.h265')]

        if h264:
            selected.append(h264[0])
            if len(h264) > 1:
                duplicates.append((ts, h264))
        elif h265:
            selected.append(h265[0])
            if len(h265) > 1:
                duplicates.append((ts, h265))
        else:
            failed.append(ts)

    return selected, duplicates, failed


def main(root_dir, timestamp_txt, output_txt):
    timestamps = extract_timestamps_from_txt(timestamp_txt)
    timestamps = list(set(timestamps))  # 去重时间戳

    print(f"📌 读取到时间戳数量: {len(timestamps)}")

    ts_to_files = find_videos_by_timestamps(root_dir, timestamps)
    selected, duplicates, failed = select_preferred_videos(ts_to_files)

    # 写成功结果
    with open(output_txt, 'w', encoding='utf-8') as f:
        for p in selected:
            f.write(p + '\n')

    print(f"\n✔ 成功找到视频: {len(selected)}")

    # 打印重复
    if duplicates:
        print("\n⚠ 发现重复视频（同一时间戳多个文件）：")
        for ts, files in duplicates:
            print(f"  时间戳 {ts}:")
            for f in files:
                print(f"    {f}")

    # 打印失败
    if failed:
        print("\n✘ 以下时间戳未找到任何 h264/h265：")
        for ts in failed:
            print(f"  {ts}")

    print("\n✅ 处理完成")


if __name__ == "__main__":
    root_dir = r"\\dtc-fs04\SmartCar_Collect\common"  
    timestamp_txt = r"C:\Users\chz62985\Desktop\dtc\新文件1.txt"
    output_txt = r"C:\Users\chz62985\Desktop\dtc\新文件11.txt"

    main(root_dir, timestamp_txt, output_txt)
