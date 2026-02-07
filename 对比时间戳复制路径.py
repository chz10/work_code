import os
from collections import defaultdict


def filter_paths_by_timestamp(path_txt, ts_txt, output_txt):
    # 1. 读取时间戳
    with open(ts_txt, 'r', encoding='utf-8') as f:
        timestamps = [line.strip() for line in f if line.strip()]

    print(f"✅ 读取时间戳数量: {len(timestamps)}")

    # path -> 命中的时间戳列表
    hit_map = defaultdict(list)

    # 2. 扫描路径文件
    with open(path_txt, 'r', encoding='utf-8') as src:
        for line in src:
            line = line.strip()
            if not line:
                continue

            for ts in timestamps:
                if ts in line:
                    hit_map[line].append(ts)

    # 3. 写入去重后的匹配路径
    with open(output_txt, 'w', encoding='utf-8') as out:
        for path in hit_map.keys():
            out.write(path + '\n')

    # # 4. 统计命中多个时间戳的“重复项”
    duplicate_items = {
        path: ts_list
        for path, ts_list in hit_map.items()
        if len(ts_list) > 1
    }

    # 5. 打印统计信息
    print(f"🎯 匹配到路径总数（去重后）: {len(hit_map)}")
    print(f"♻️  命中多个时间戳的路径数量: {len(duplicate_items)}")

    if duplicate_items:
        print("\n📌 命中多个时间戳的路径明细：")
        for path, ts_list in duplicate_items.items():
            print(path)
            print(f"   ➜ 命中 {len(ts_list)} 次: {', '.join(ts_list)}")

    print(f"\n📄 结果文件已保存到: {output_txt}")


if __name__ == "__main__":
    path_txt = r"C:\Users\chz62985\Desktop\素管素材.txt"
    ts_txt = r"C:\Users\chz62985\Desktop\新建 文本文档 (2).txt"

    # ⚠️ 建议不要和 ts_txt 同名，避免覆盖
    output_txt = r"C:\Users\chz62985\Desktop\dwz_lixiang2.txt"

    filter_paths_by_timestamp(path_txt, ts_txt, output_txt)
