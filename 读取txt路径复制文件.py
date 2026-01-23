import os
import shutil
from tqdm import tqdm


def read_txt_list(txt_path):
    """读取 txt，每行一个文件路径"""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"txt 不存在: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def copy_files_with_progress(file_list, output_dir, use_subfolder=False):
    """
    根据路径复制单个文件
    - 不限制文件类型
    - 显示进度
    - 可选：为每个文件创建同名文件夹
    """
    os.makedirs(output_dir, exist_ok=True)

    success = 0
    failed = 0

    for src_path in tqdm(file_list, desc="📦 正在复制文件", unit="file"):
        try:
            if not os.path.isfile(src_path):
                raise FileNotFoundError("源文件不存在")

            file_name = os.path.basename(src_path)
            name_no_ext, _ = os.path.splitext(file_name)

            if use_subfolder:
                target_dir = os.path.join(output_dir, name_no_ext)
                os.makedirs(target_dir, exist_ok=True)
                dst_path = os.path.join(target_dir, file_name)
            else:
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

    txt_path = r"C:\Users\chz62985\Desktop\新建 文本文档 (2).txt"
    output_dir = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_80\input\fangyeqing\20260113"

    
    # === 新增：是否创建同名文件夹 ===
    while True:
        user_input = input("是否为每个文件创建同名文件夹？(yes/y/no): ").strip().lower()
        if user_input in ("yes", "y"):
            use_subfolder = True
            break
        elif user_input in ("no", "n", ""):
            use_subfolder = False
            break
        else:
            print("⚠️ 请输入 yes / no")


    file_list = read_txt_list(txt_path)
    copy_files_with_progress(file_list, output_dir, use_subfolder)
