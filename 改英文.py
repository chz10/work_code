import os
import re
from pypinyin import lazy_pinyin

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def chinese_to_pinyin(text):
    return "_".join(lazy_pinyin(text))

def safe_rename(old, new):
    try:
        os.rename(old, new)
        print(f"✅ 重命名成功: {old} -> {new}")
    except Exception as e:
        print(f"❌ 重命名失败: {old}")
        print(f"   错误原因: {e}")

def rename_dirs_and_files(root_path):
    for root, dirs, files in os.walk(root_path, topdown=False):

        # ---------- 文件 ----------
        for filename in files:
            if contains_chinese(filename):
                old_path = os.path.join(root, filename)
                name, ext = os.path.splitext(filename)
                new_name = chinese_to_pinyin(name) + ext
                new_path = os.path.join(root, new_name)

                if old_path != new_path:
                    safe_rename(old_path, new_path)

        # ---------- 文件夹 ----------
        for d in dirs:
            if contains_chinese(d):
                old_dir = os.path.join(root, d)
                new_dir = os.path.join(root, chinese_to_pinyin(d))

                if old_dir != new_dir:
                    safe_rename(old_dir, new_dir)

if __name__ == "__main__":
    ROOT_DIR = r"\\?\UNC\hz-iotfs02\Model_Test\TestSpace\Personal_Space\DWZ\test_data\FT\TLR"
    rename_dirs_and_files(ROOT_DIR)
