import os
from collections import defaultdict

# ----------------------------------------------------------------------
# 路径映射规则（共享路径 → 本地路径）
# ----------------------------------------------------------------------
PATH_MAPPING = {
    r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space": "/tmp/iot_test/mnt_data",
    r"\\Material": "/tmp/iot_test/mnt_data",
}

VIDEO_SUFFIX = (".h264", ".h265")


# ----------------------------------------------------------------------
# 路径规范化
# ----------------------------------------------------------------------
def normalize_path(file_path: str) -> str:
    for src, dst in PATH_MAPPING.items():
        if src in file_path:
            file_path = file_path.replace(src, dst)
            break
    return file_path.replace("\\", "/")


# ----------------------------------------------------------------------
# 车型识别（强规则 + 兜底）
# ----------------------------------------------------------------------
def extract_car_type(path: str) -> str:
    parts = path.replace("\\", "/").split("/")

    # ---------- 强规则：xin_data ----------
    for i, part in enumerate(parts):
        if part.startswith("xin_data_") and i >= 1:
            return parts[i - 1]

    # ---------- 兜底规则 ----------
    candidates = []
    for p in parts:
        pl = p.lower()
        if (
            3 <= len(p) <= 20
            and not p.isdigit()
            and any(c.isalpha() for c in p)
            and not pl.startswith(("xin", "video", "output"))
        ):
            candidates.append(p)

    return candidates[-1] if candidates else "unknown"


# ----------------------------------------------------------------------
# 收集视频文件（按文件名分组）
# ----------------------------------------------------------------------
def collect_video_files(src_path: str):
    file_map = defaultdict(list)

    for root, _, files in os.walk(src_path):
        for name in files:
            if not name.lower().endswith(VIDEO_SUFFIX):
                continue

            full_path = os.path.join(root, name)
            full_path = normalize_path(full_path)
            file_map[name].append(full_path)

    return file_map


# ----------------------------------------------------------------------
# 同名文件去重规则
# ----------------------------------------------------------------------
def select_best_path(paths):
    non_bu = [p for p in paths if "_bu" not in p]
    return non_bu[0] if non_bu else paths[0]


# ----------------------------------------------------------------------
# 主逻辑
# ----------------------------------------------------------------------
def main():
    src_video_path = input("请输入源视频路径: ").strip()
    output_txt = input("请输入输出 txt 路径(如 C:\\\\xxx\\\\out.txt): ").strip()

    base_dir = os.path.dirname(output_txt)
    os.makedirs(base_dir, exist_ok=True)

    file_map = collect_video_files(src_video_path)

    # 按车型聚合
    car_map = defaultdict(list)

    for filename, paths in file_map.items():
        best_path = select_best_path(paths)
        car_type = extract_car_type(best_path)
        car_map[car_type].append((filename, best_path))

    # 输出 all.txt
    all_txt = os.path.join(base_dir, "all.txt")
    with open(all_txt, "w", encoding="utf-8") as f:
        for car, items in sorted(car_map.items()):
            for name, path in items:
                f.write(f"{name}\t{path}\n")

    # 按车型输出
    for car, items in sorted(car_map.items()):
        car_txt = os.path.join(base_dir, f"{car}.txt")
        with open(car_txt, "w", encoding="utf-8") as f:
            for name, path in items:
                f.write(f"{name}\t{path}\n")

    # 统计信息
    print("\n========== 统计结果 ==========")
    total = 0
    for car, items in sorted(car_map.items()):
        print(f"{car:10s} : {len(items)}")
        total += len(items)

    print(f"TOTAL      : {total}")
    print(f"\n✅ 完成！输出目录：{base_dir}")


if __name__ == "__main__":
    main()
