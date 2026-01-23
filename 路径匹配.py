import os

def filter_paths_by_timestamp(path_txt, ts_txt, output_txt):
    # 1️⃣ 读取 ts（严格保持原始顺序）
    with open(ts_txt, 'r', encoding='utf-8') as f:
        timestamps = [line.strip() for line in f if line.strip()]

    print(f"✅ 读取时间戳数量: {len(timestamps)}")

    # 2️⃣ 读取所有路径
    with open(path_txt, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"📂 读取路径数量: {len(paths)}")

    unmatched_list = []

    # 3️⃣ 核心逻辑：一条 ts → 一条输出（不多不少）
    with open(output_txt, 'w', encoding='utf-8') as out:
        for ts in timestamps:
            matched_path = None

            for path in paths:
                if ts in path:
                    matched_path = path
                    break   # ✅ 只取第一条匹配

            if matched_path:
                out.write(f"{ts} | {matched_path}\n")
            else:
                out.write(f"{ts} | <NO_MATCH>\n")
                unmatched_list.append(ts)

    # 4️⃣ 统计结果
    print("\n========== 统计结果 ==========")
    print(f"🧾 输出总行数: {len(timestamps)}")
    print(f"❌ 未命中时间戳数量: {len(unmatched_list)}")

    if unmatched_list:
        print("\n📌 未命中时间戳示例（最多显示 20 个）：")
        for ts in unmatched_list[:20]:
            print(f"  - {ts}")

    print(f"\n📄 结果文件已保存到: {output_txt}")


if __name__ == "__main__":
    path_txt = r"C:\Users\chz62985\Desktop\素管素材.txt"
    ts_txt   = r"C:\Users\chz62985\Desktop\时间戳.txt"

    # ⚠️ 一定不要和 ts_txt 同名
    output_txt = r"C:\Users\chz62985\Desktop\路径匹配结果.txt"

    filter_paths_by_timestamp(path_txt, ts_txt, output_txt)
