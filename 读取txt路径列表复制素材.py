import os
import shutil
import pandas as pd

def safe_copy_folder(source, dest, log_path="copy_failed_log.txt"):
    os.makedirs(dest, exist_ok=True)
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        for root, _, files in os.walk(source):
            rel_path = os.path.relpath(root, source)
            dest_root = os.path.join(dest, rel_path)
            os.makedirs(dest_root, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_root, file)
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    log_msg = f"❌ 跳过文件: {src_file} -> 错误: {str(e)}\n"
                    print(log_msg.strip())
                    log_file.write(log_msg)


def save_folder_to_nested_folders(source_folder_path, base_dir, log_path="copy_failed_log.txt"):
    # 构建嵌套路径
    folder_path = os.path.join(
        base_dir,
        # file_dict['更新问题'],
    )

    os.makedirs(folder_path, exist_ok=True)

    # 保持源文件夹名
    folder_name = os.path.basename(source_folder_path.rstrip(os.sep))
    dest_path = os.path.join(folder_path, folder_name)

    # 检查是否已存在
    if os.path.exists(dest_path):
        print(f"⚠️ 目标文件夹已存在，跳过复制: {dest_path}")
    else:
        try:
            shutil.copytree(source_folder_path, dest_path)
            print(f"✅ 文件夹已复制到: {dest_path}")
        except Exception as e:
            print(f"⚠️ copytree 失败: {e}，使用逐文件复制 fallback")
            safe_copy_folder(source_folder_path, dest_path, log_path=log_path)



def read_txt_list(txt_path):
    """读取txt文件，将每行的路径存入列表（去除空行和换行符）"""
    file_list = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"文件不存在：{txt_path}")

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            path = line.strip()  # 去除换行和首尾空格
            if path:  # 跳过空行
                file_list.append(path)

    return file_list

# def save_file_to_nested_folders(file_dict, source_file_path, base_dir, log_path="copy_failed_log.txt"):
#     # 构建嵌套路径（根据 file_dict 信息）
#     folder_path = os.path.join(
#         base_dir,
#         # 这里可以替换为你需要的层级，比如：
#         # file_dict["category"], file_dict["date"], ...
#     )

#     os.makedirs(folder_path, exist_ok=True)

#     # 保持源文件名
#     file_name = os.path.basename(source_file_path)
#     dest_path = os.path.join(folder_path, file_name)

#     # 检查是否已存在
#     if os.path.exists(dest_path):
#         print(f"⚠️ 目标文件已存在，跳过复制: {dest_path}")
#     else:
#         try:
#             shutil.copy2(source_file_path, dest_path)  # copy2 会保留元信息（时间戳等）
#             print(f"✅ 文件已复制到: {dest_path}")
#         except Exception as e:
#             print(f"❌ 文件复制失败: {e}")
#             with open(log_path, "a", encoding="utf-8") as f:
#                 f.write(f"文件复制失败: {source_file_path} -> {dest_path}, 错误: {e}\n")


if __name__ == "__main__":

    txt_path = r"C:\Users\chz62985\Desktop\jira (2).txt"
    output_dir = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_80\input\buzongtao\20260127y"
    h264_list = read_txt_list(txt_path)
    for h264_path in h264_list:
        if "xin" in h264_path:
            # h264_dir = os.path.dirname(os.path.dirname(h264_path))
            h264_dir = os.path.dirname(h264_path)
        elif "VisInsight" in h264_path:
            # h264_dir = os.path.dirname(h264_path)
            h264_dir = h264_path
        else:
            print(f"数据源未匹配，{h264_path}")
            continue

        save_folder_to_nested_folders(h264_dir, output_dir)

