
import os
from collections import defaultdict

# ----------------------------------------------------------------------
# 路径映射规则
# ----------------------------------------------------------------------
PATH_MAPPING = {
    r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space": "/tmp/iot_test/mnt_data",
    r"\\Material\xuekangkang\download": "/tmp/iot_test/mnt_data",
}

VIDEO_SUFFIX = (".h264", ".h265")


def normalize_path(file_path: str) -> str:
    """统一路径格式并做共享路径替换"""
    for src, dst in PATH_MAPPING.items():
        if src in file_path:
            file_path = file_path.replace(src, dst)
            break
    return file_path.replace("\\", "/")


def collect_video_files(src_path: str):
    """
    遍历目录，按文件名分组收集视频路径
    return: dict { filename: [path1, path2, ...] }
    """
    file_map = defaultdict(list)

    for root, _, files in os.walk(src_path):
        for name in sorted(files):
            if not name.lower().endswith(VIDEO_SUFFIX):
                continue

            full_path = os.path.join(root, name)
            full_path = normalize_path(full_path)

            file_map[name].append(full_path)

    return file_map


def select_best_path(paths):
    """
    同一文件名下：
    - 优先选择不含 '_bu' 的路径
    - 否则保留第一个
    """
    non_bu = [p for p in paths if "_bu" not in p]
    return non_bu[0] if non_bu else paths[0]


def write_dedup_result(file_map, out_fp):
    """去重并写入结果"""
    count = 0
    for filename, paths in sorted(file_map.items()):
        best_path = select_best_path(paths)
        out_fp.write(best_path + "\n")
        print(best_path)
        count += 1

    return count


def main():
    output_txt = r"C:\Users\chz62985\Desktop\gzy素材.txt"

    src_video_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\GZY"

    file_map = collect_video_files(src_video_path)

    with open(output_txt, "w", encoding="utf-8") as fp:
        count = write_dedup_result(file_map, fp)

    print(f"✅ 视频路径提取完成，共写入 {count} 个（已优先过滤 _bu）")


if __name__ == "__main__":
    main()
