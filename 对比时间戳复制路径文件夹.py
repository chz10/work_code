import os
import re
from collections import defaultdict


def load_timestamps(ts_txt):
    """
    读取时间戳并生成正则
    """
    with open(ts_txt, 'r', encoding='utf-8') as f:
        ts_list = [line.strip() for line in f if line.strip()]

    if not ts_list:
        return [], None

    # ⚡ 合并成一个正则：ts1|ts2|ts3
    pattern = re.compile("|".join(map(re.escape, ts_list)))
    return ts_list, pattern


def filter_one_pair(path_txt, ts_txt, output_txt):
    ts_list, ts_pattern = load_timestamps(ts_txt)

    if not ts_list:
        print(f"⚠️ 空时间戳文件 | {os.path.basename(ts_txt)}")
        return

    hit_map = defaultdict(list)

    # 只扫一遍 path 文件
    with open(path_txt, 'r', encoding='utf-8') as src:
        for line in src:
            line = line.strip()
            if not line:
                continue

            # ⚡ 一次正则查找
            matches = ts_pattern.findall(line)
            if matches:
                hit_map[line].extend(set(matches))

    # ❌ 没有任何命中，不生成文件
    if not hit_map:
        print(f"❌ 无匹配 | {os.path.basename(path_txt)} × {os.path.basename(ts_txt)}")
        return

    # ✅ 只有命中才写文件
    with open(output_txt, 'w', encoding='utf-8') as out:
        for path in hit_map.keys():
            out.write(path + '\n')

    # ♻️ 命中多个时间戳
    duplicate_items = {
        path: ts
        for path, ts in hit_map.items()
        if len(ts) > 1
    }

    print(f"✅ 命中 | {os.path.basename(path_txt)} × {os.path.basename(ts_txt)}")
    print(f"   🎯 匹配路径数: {len(hit_map)}")

    if duplicate_items:
        print(f"   ♻️ 多时间戳命中: {len(duplicate_items)}")
        for path, ts in duplicate_items.items():
            print(f"      {path}")
            print(f"         ➜ {', '.join(ts)}")

    print(f"   📄 输出文件: {output_txt}\n")


def batch_match(path_dir, ts_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    path_files = [
        os.path.join(path_dir, f)
        for f in os.listdir(path_dir)
        if f.lower().endswith(".txt")
    ]

    ts_files = [
        os.path.join(ts_dir, f)
        for f in os.listdir(ts_dir)
        if f.lower().endswith(".txt")
    ]

    print(f"📂 path_txt 数量: {len(path_files)}")
    print(f"📂 ts_txt   数量: {len(ts_files)}\n")

    for path_txt in path_files:
        path_name = os.path.splitext(os.path.basename(path_txt))[0]

        for ts_txt in ts_files:
            ts_name = os.path.splitext(os.path.basename(ts_txt))[0]

            output_txt = os.path.join(
                output_dir,
                f"{ts_name}.txt"
            )

            filter_one_pair(path_txt, ts_txt, output_txt)



if __name__ == "__main__":
    path_dir = r"C:\Users\chz62985\Desktop\gzy\fagui"
    ts_dir = r"C:\Users\chz62985\Desktop\gzy\out"
    output_dir = r"C:\Users\chz62985\Desktop\gzy\out1"

    batch_match(path_dir, ts_dir, output_dir)
