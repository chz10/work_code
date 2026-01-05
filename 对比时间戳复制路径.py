import os

def filter_paths_by_timestamp(path_txt, ts_txt, output_txt):
    # 读取时间戳，放入 set（查找速度极快）
    with open(ts_txt, 'r', encoding='utf-8') as f:
        timestamps = set(line.strip() for line in f if line.strip())

    print(f"✅ 读取时间戳数量: {len(timestamps)}")

    matched = 0
    with open(path_txt, 'r', encoding='utf-8') as src, \
         open(output_txt, 'w', encoding='utf-8') as out:

        for line in src:
            line = line.strip()
            if not line:
                continue

            # 判断是否包含任一时间戳
            for ts in timestamps:
                if ts in line:
                    out.write(line + '\n')
                    matched += 1
                    break

    print(f"🎯 匹配到路径数量: {matched}")
    print(f"📄 结果已保存到: {output_txt}")
if __name__ == "__main__":
    path_txt = r"C:\Users\chz62985\Desktop\dwz\111111.txt"
    ts_txt = r"C:\Users\chz62985\Desktop\dwz\lixiang3.txt"
    output_txt = r"C:\Users\chz62985\Desktop\dwz\xin_lixiang3.txt"

    filter_paths_by_timestamp(path_txt, ts_txt, output_txt)

