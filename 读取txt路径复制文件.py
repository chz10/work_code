import os
import shutil
from tqdm import tqdm


def read_txt_list(txt_path):
    """读取 txt，每行一个文件路径"""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"txt 不存在: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def copy_files_with_progress(file_list, output_dir):
    """
    根据路径复制单个文件
    - 不限制文件类型
    - 显示进度
    - 失败直接终端输出
    """
    os.makedirs(output_dir, exist_ok=True)

    success = 0
    failed = 0

    for src_path in tqdm(file_list, desc="📦 正在复制文件", unit="file"):
        try:
            if not os.path.isfile(src_path):
                raise FileNotFoundError("源文件不存在")

            file_name = os.path.basename(src_path)
            dst_path = os.path.join(output_dir, file_name)

            if os.path.exists(dst_path):
                print(f"⚠️ 已存在，跳过: {dst_path}")
                continue

            shutil.copy2(src_path, dst_path)
            success += 1

        except Exception as e:
            failed += 1
            print(f"\n❌ 复制失败: {src_path}")
            print(f"   错误原因: {e}")

    print("\n========== 复制完成 ==========")
    print(f"✅ 成功: {success}")
    print(f"❌ 失败: {failed}")
    print(f"📁 目标目录: {output_dir}")


if __name__ == "__main__":

    txt_path = r"C:\Users\chz62985\Desktop\新建 文本文档 (4).txt"
    output_dir = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\GZY\latdistance\hq1"

    file_list = read_txt_list(txt_path)
    copy_files_with_progress(file_list, output_dir)
