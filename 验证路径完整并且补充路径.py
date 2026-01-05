import os

def complete_h264_paths(
    input_txt,
    output_txt,
    fail_txt
):
    completed = []
    failed = []

    with open(input_txt, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]

    for path in paths:
        # 情况 1：已经是 .h264 文件
        if path.lower().endswith('.h264'):
            if os.path.isfile(path):
                completed.append(path)
            else:
                failed.append(f"[文件不存在] {path}")
            continue

        # 情况 2：是目录（不完整路径）
        if not os.path.isdir(path):
            failed.append(f"[目录不存在] {path}")
            continue

        video_dir = os.path.join(path, 'video')
        if not os.path.isdir(video_dir):
            failed.append(f"[无 video 目录] {path}")
            continue

        h264_files = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith('.h264')
        ]

        if len(h264_files) == 0:
            failed.append(f"[video 下无 h264] {path}")
        elif len(h264_files) == 1:
            completed.append(h264_files[0])
        else:
            # 多个 h264，全部列出，避免误判
            completed.extend(h264_files)

    # 写成功结果
    with open(output_txt, 'w', encoding='utf-8') as f:
        for p in completed:
            f.write(p + '\n')

    # 写失败日志
    with open(fail_txt, 'w', encoding='utf-8') as f:
        for p in failed:
            f.write(p + '\n')

    print(f"完成路径数: {len(completed)}")
    print(f"失败路径数: {len(failed)}")


if __name__ == "__main__":
    input_txt = r"C:\Users\chz62985\Desktop\liuyang\FT.txt"
    output_txt = r"C:\Users\chz62985\Desktop\liuyang\FT2.txt"
    fail_txt = r"C:\Users\chz62985\Desktop\liuyang\FT2failed_paths.txt"

    complete_h264_paths(input_txt, output_txt, fail_txt)
