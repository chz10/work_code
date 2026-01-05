import os


def find_specified_files(folder_path, file_extension=".h264"):
        search_results = {}
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(file_extension):
                    filename = os.path.splitext(file)[0]
                    if filename not in search_results:
                        search_results[filename] = os.path.join(root, file)
        return search_results


if __name__ == "__main__":
    video_path = "\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_50\input\gaoziyi\20251217\lixiang1_1"
    h264_files = find_specified_files(video_path)

    recharge_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_50\input\gaoziyi\20251217\lixiang1_1"
    recharge_files = find_specified_files(recharge_path, file_extension=".txt")
    for filename, filepath in h264_files.items():
        if filename in recharge_files:
            continue
        else:
            print(f"缺少对应txt文件: {filename}")

    # missing_count = 0  # 新增统计变量

    # for filename, filepath in h264_files.items():
    #     if filename in recharge_files:
    #         continue
    #     else:
    #         print(f"缺少对应txt文件: {filename}")
    #         missing_count += 1  # 计数

    # print(f"总共缺少 {missing_count} 个 txt 文件")

