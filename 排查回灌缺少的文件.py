import os

def find_specified_files(folder_path, file_extension=".h264"):
    """
    查找指定目录下的文件
    
    Args:
        folder_path: 要搜索的目录路径
        file_extension: 文件扩展名（如'.h264'或'.txt'）
    
    Returns:
        dict: {文件名: 完整路径} 的字典
    """
    search_results = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                filename = os.path.splitext(file)[0]
                if filename not in search_results:
                    search_results[filename] = os.path.join(root, file)
    return search_results

def main():
    # 设置路径
    video_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_new\input\xiefengfan\20251209\lixiang2"
    recharge_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_new\output\xiefengfan\20251209\lixiang2\output"
    
    # 获取文件列表
    h264_files = find_specified_files(video_path)
    recharge_files = find_specified_files(recharge_path, file_extension=".txt")
    
    # 找出缺少的文件
    missing_files = []
    for filename in h264_files:
        if filename not in recharge_files:
            missing_files.append(filename)
    
    # 输出结果
    if missing_files:
        # print(f"总共缺少 {len(missing_files)} 个 txt 文件：")py
        for filename in missing_files:
            print(filename)
    else:
        print("所有文件都有对应的txt文件")

    print(f"总共缺少 {len(missing_files)} 个 txt 文件：")

if __name__ == "__main__":
    main()
