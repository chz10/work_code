import re
import os
from datetime import datetime

# ================== 配置区 ==================

# 输入 txt（存放所有路径）
INPUT_TXT = r"C:\Users\chz62985\Desktop\lixiang2.txt"

# 输出目录
OUTPUT_DIR = r"C:\Users\chz62985\Desktop\新建文件夹"

# 三个时间区间
RANGES = {
    "A_20250824_20250922": (
        datetime(2025, 8, 24),
        datetime(2025, 9, 22)
    ),
    "B_20250923_20251009": (
        datetime(2025, 9, 23),
        datetime(2025, 10, 9)
    ),
    "C_20251010_20260101": (
        datetime(2025, 10, 10),
        datetime(2026, 1, 1)
    )
}

# ================== 正则 ==================
# 匹配 14 位时间戳，后缀是 .h264 或 .h265
TIMESTAMP_PATTERN = re.compile(r'(\d{14})(?=\.h26[45])')

# ================== 主逻辑 ==================

def classify_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 打开输出文件
    output_files = {
        key: open(os.path.join(OUTPUT_DIR, f"{key}.txt"), "w", encoding="utf-8")
        for key in RANGES
    }

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue

            match = TIMESTAMP_PATTERN.search(path)
            if not match:
                continue

            timestamp = match.group(1)
            date_str = timestamp[:8]  # YYYYMMDD

            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                continue

            for key, (start, end) in RANGES.items():
                if start <= date_obj <= end:
                    output_files[key].write(path + "\n")
                    break

    for f in output_files.values():
        f.close()

    print("分类完成，结果已输出到：", OUTPUT_DIR)


if __name__ == "__main__":
    classify_paths()
