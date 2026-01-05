import os
import re


def clean_path(path: str) -> str:
    """去掉路径首尾的引号和空格"""
    return path.strip().strip('"').strip("'")


def extract_timestamps_from_path(path):
    """从整条路径中提取所有 14 位时间戳"""
    return re.findall(r'\d{14}', path)


def find_video_by_timestamp(root_dir, timestamps):
    """
    在 root_dir 下递归查找：
    1️⃣ 优先返回匹配时间戳的 .h264
    2️⃣ 若不存在，再返回匹配时间戳的 .h265
    """
    h264_matches = []
    h265_matches = []

    for root, _, files in os.walk(root_dir):
        for f in files:
            lower_f = f.lower()
            if not (lower_f.endswith('.h264') or lower_f.endswith('.h265')):
                continue

            for ts in timestamps:
                if ts in f:
                    full_path = os.path.join(root, f)
                    if lower_f.endswith('.h264'):
                        h264_matches.append(full_path)
                    else:
                        h265_matches.append(full_path)
                    break

    if h264_matches:
        return h264_matches, 'h264'
    if h265_matches:
        return h265_matches, 'h265'
    return [], None


def try_dir_from_video_path(video_path):
    """
    当 xxx.h264 不存在时：
    尝试把 xxx 当目录
    """
    base = re.sub(r'\.(h264|h265)$', '', video_path, flags=re.IGNORECASE)
    return base if os.path.isdir(base) else None


def check_and_complete_paths(input_txt, output_txt, fail_txt):
    completed = []
    failed = []

    with open(input_txt, 'r', encoding='utf-8') as f:
        raw_paths = [line.strip() for line in f if line.strip()]

    for raw_path in raw_paths:
        path = clean_path(raw_path)

        # ========= 情况 1：路径看起来是视频文件 =========
        if path.lower().endswith(('.h264', '.h265')):
            if os.path.isfile(path):
                completed.append(path)
                continue

            # 尝试：去掉后缀当目录
            possible_dir = try_dir_from_video_path(path)
            if possible_dir:
                timestamps = extract_timestamps_from_path(path)
                video_files, _ = find_video_by_timestamp(possible_dir, timestamps)
                if video_files:
                    completed.extend(video_files)
                    continue

            failed.append(f"[视频文件不存在] {path}")
            continue

        # ========= 情况 2：目录路径 =========
        if not os.path.isdir(path):
            failed.append(f"[目录不存在] {path}")
            continue

        timestamps = extract_timestamps_from_path(path)
        if not timestamps:
            failed.append(f"[路径中未发现 14 位时间戳] {path}")
            continue

        video_files, _ = find_video_by_timestamp(path, timestamps)
        if not video_files:
            failed.append(f"[未找到匹配 h264/h265] {path}")
        else:
            completed.extend(video_files)

    # 写成功结果
    with open(output_txt, 'w', encoding='utf-8') as f:
        for p in completed:
            f.write(p + '\n')

    # 写失败结果
    with open(fail_txt, 'w', encoding='utf-8') as f:
        for p in failed:
            f.write(p + '\n')

    print(f"✔ 成功补全视频文件: {len(completed)}")
    print(f"✘ 失败路径数: {len(failed)}")


if __name__ == "__main__":
    input_txt = r"C:\Users\chz62985\Desktop\新建 文本文档.txt"
    output_txt = r"C:\Users\chz62985\Desktop\新建 文本文档1.txt"
    fail_txt = r"C:\Users\chz62985\Desktop\新建 文本文档2.txt"

    check_and_complete_paths(input_txt, output_txt, fail_txt)
    # check_and_complete_paths(input_txt, output_txt)



# if __name__ == "__main__":
#     input_txt = r"C:\Users\chz62985\Desktop\lixiang2.txt"
#     output_txt = r"C:\Users\chz62985\Desktop\liuyang\lixiang2.txt"
#     fail_txt = r"C:\Users\chz62985\Desktop\liuyang\lixiang2failed_paths.txt"
#
#     check_and_complete_paths(input_txt, output_txt, fail_txt)
