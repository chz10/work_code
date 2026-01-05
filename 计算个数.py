import os

def count_txt_files(root_path):
    count = 0
    print("开始遍历目录：", root_path)
    print("-" * 60)

    for dirpath, dirnames, filenames in os.walk(root_path):
        print(f"\n正在扫描目录: {dirpath}")
        
        for filename in filenames:
            print(f"    文件: {filename}")
            if filename.endswith(".txt"):  # 查找 txt 文件
                count += 1
                print(f"     匹配到以 '.txt' 结尾的文件: {filename}")

    print("-" * 60)
    return count


root_directory = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_50\output_test\gaoziyi\V3.1_2M_3.1.27223.1350_MSIND"
result = count_txt_files(root_directory)
print(f"\n最终统计结果：扩展名为 '.txt' 的文件数量为：{result}")



# def count_xin_dirs(root_path):
#     count = 0
#     print("开始遍历目录：", root_path)
#     print("-" * 60)
#
#     for dirpath, dirnames, filenames in os.walk(root_path):
#         print(f"\n正在扫描目录: {dirpath}")
#
#         for dirname in dirnames:
#             print(f"    文件夹: {dirname}")
#             if dirname.startswith("xin"):  # 检查是否以'xin'开头
#                 count += 1
#                 print(f"     匹配到以 'xin' 开头的文件夹: {dirname}")
#
#     print("-" * 60)
#     return count
#
# # 查找文件路径
# root_directory = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_50\input\duwenzhe\20251216"
# result = count_xin_dirs(root_directory)
# print(f"\n最终统计结果：以 'xin' 开头的文件夹数量为：{result}")
