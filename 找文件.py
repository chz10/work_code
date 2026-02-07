import os

VIDEO_SUFFIX = (".h264", ".h265")


def find_videos(root_dir, output_txt):
    root_dir = root_dir.strip().strip('"').strip("'")

    video_paths = []

    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(VIDEO_SUFFIX):
                full_path = os.path.join(root, name)
                video_paths.append(full_path)

    # 写入 txt
    with open(output_txt, "w", encoding="utf-8") as f:
        for path in video_paths:
            f.write(path + "\n")

    print(f"✅ 查找完成，共找到 {len(video_paths)} 个视频文件")
    print(f"📄 已保存到：{output_txt}")


if __name__ == "__main__":
    # 👉 把这里换成你的网络路径
    ROOT_PATH = r"\\dtc-fs04\SmartCar_Collect\common\ft_2m_geely_ss21_8004\20260127"

    # 👉 输出的 txt 文件名
    OUTPUT_TXT = r"C:\Users\chz62985\Desktop\fyq\geely_2239.txt"

    find_videos(ROOT_PATH, OUTPUT_TXT)
