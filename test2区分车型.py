import os
from collections import defaultdict

# ==================================================
# 视频后缀
# ==================================================
VIDEO_SUFFIX = (".h264", ".h265")

# ==================================================
# 路径映射（共享路径 → 本地路径）
# ==================================================
PATH_MAPPING = {
    r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space": "/tmp/iot_test/mnt_data",
    r"\\Material\xuekangkang\download": "/tmp/iot_test/mnt_data",
    r"\\hz-iotfs02\Function_Test\Front_Camera": "/tmp/iot_test/mnt_data",
}


def normalize_path(p: str) -> str:
    """统一路径格式"""
    for src, dst in PATH_MAPPING.items():
        if src in p:
            p = p.replace(src, dst)
            break
    return p.replace("\\", "/")


# ==================================================
# 车型关键词（核心配置）
# 只要路径中“包含”key，就认为是该车型
# ==================================================
CAR_KEYWORDS = {
    "lixiang3": "lixiang3",
    "lixiang2": "lixiang2",
    "lixiang1": "lixiang1",
    "lx3": "lixiang3",
    "lx2": "lixiang2",
    "lx1": "lixiang1",
    "lixinag1": "lixiang1",
    "lixinag2": "lixiang2",

    "natie3": "natie3",
    "natie2": "natie2",
    "nt3": "natie3",
    "nt2": "natie2",

    "Wuling_5577": "wuling_5577",
    "Wuling_5741": "wuling_5741",
    "wuling_5577": "wuling_5577",
    "wuling_5741": "wuling_5741",

    "lynkco": "lyncko",
    "lyncko": "lyncko",

    # "Geely": "geely",
    "Geely_2239": "geely_2239",
    "geely_2239": "geely_2239",
    "Geely_2506": "geely_2506",
    "geely_2506": "geely_2506",

    "gl8": "GL8",
    "GL8": "gl8",

    "hq": "HQ",
}

# 👉 防止短词抢命中（很关键）
CAR_KEYWORDS = dict(
    sorted(CAR_KEYWORDS.items(), key=lambda x: -len(x[0]))
)


# ==================================================
# 核心：提取车型（不依赖目录结构）
# ==================================================
def extract_car_type(path: str) -> str:
    parts = path.replace("\\", "/").lower().split("/")

    # 从后往前扫，越靠近文件的优先级越高
    for p in reversed(parts):
        for key, car in CAR_KEYWORDS.items():
            if key in p:
                return car

    return "unknown"


# ==================================================
# 收集视频文件（按文件名去重）
# ==================================================
def collect_video_files(src_root: str):
    file_map = defaultdict(list)

    for root, _, files in os.walk(src_root):
        for name in files:
            if not name.lower().endswith(VIDEO_SUFFIX):
                continue

            full_path = normalize_path(os.path.join(root, name))
            file_map[name].append(full_path)

    return file_map


def select_best_path(paths):
    """
    同名视频多路径时的选择策略
    优先选不带 _bu 的
    """
    non_bu = [p for p in paths if "_bu" not in p.lower()]
    return non_bu[0] if non_bu else paths[0]


# ==================================================
# 主程序
# ==================================================
def main():
    src_root = input("请输入源视频路径: ").strip()
    out_dir = input("请输入输出目录: ").strip()

    os.makedirs(out_dir, exist_ok=True)

    file_map = collect_video_files(src_root)
    car_map = defaultdict(list)
    unknown_list = []

    for name, paths in file_map.items():
        best_path = select_best_path(paths)
        car = extract_car_type(best_path)

        if car == "unknown":
            unknown_list.append(best_path)
        else:
            car_map[car].append(best_path)

    # 每个车型一个 txt
    for car, paths in sorted(car_map.items()):
        with open(os.path.join(out_dir, f"xkk_{car}.txt"), "w", encoding="utf-8") as f:
            for p in paths:
                f.write(p + "\n")

    # unknown 单独输出，方便你补关键词
    if unknown_list:
        with open(os.path.join(out_dir, "error_unknown.txt"), "w", encoding="utf-8") as f:
            for p in unknown_list:
                f.write(p + "\n")

    # 统计结果
    print("\n========== 统计结果 ==========")
    total = 0
    for car, paths in sorted(car_map.items()):
        print(f"{car:15s}: {len(paths)}")
        total += len(paths)

    if unknown_list:  
        print(f"{'unknown':15s}: {len(unknown_list)}")
        total += len(unknown_list)

    print(f"{'TOTAL':15s}: {total}")
    print(f"\n✅ 完成！输出目录：{out_dir}")


if __name__ == "__main__":
    main()
