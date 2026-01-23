import os
import re
from pypinyin import lazy_pinyin

def contains_chinese(text):
    """判断是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def chinese_to_pinyin(text):
    """中文转拼音，用下划线连接"""
    return "_".join(lazy_pinyin(text))

def rename_dirs_and_files(root_path):
    """
    递归重命名：
    - 中文文件夹 → 拼音
    - 中文文件名 → 拼音（保留后缀）
    """
    for root, dirs, files in os.walk(root_path, topdown=False):

        # ---------- 先改文件 ----------
        for filename in files:
            if contains_chinese(filename):
                old_path = os.path.join(root, filename)

                name, ext = os.path.splitext(filename)
                new_name = chinese_to_pinyin(name) + ext
                new_path = os.path.join(root, new_name)

                if not os.path.exists(new_path):
                    print(f"文件: {old_path} -> {new_path}")
                    os.rename(old_path, new_path)
                else:
                    print(f"⚠ 文件已存在，跳过: {new_path}")

        # ---------- 再改文件夹 ----------
        for d in dirs:
            if contains_chinese(d):
                old_dir = os.path.join(root, d)
                new_dir_name = chinese_to_pinyin(d)
                new_dir = os.path.join(root, new_dir_name)

                if not os.path.exists(new_dir):
                    print(f"目录: {old_dir} -> {new_dir}")
                    os.rename(old_dir, new_dir)
                else:
                    print(f"⚠ 目录已存在，跳过: {new_dir}")

if __name__ == "__main__":
    ROOT_DIR = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\XKK\fixed_scene"
    rename_dirs_and_files(ROOT_DIR)
