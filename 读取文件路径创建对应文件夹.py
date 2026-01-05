import os
import shutil

def copy_files_from_txt(txt_path, output_root):
    """
    从 txt 中读取文件路径，
    每个文件创建一个同名文件夹并复制进去
    """
    if not os.path.isfile(txt_path):
        print(f"❌ txt 文件不存在: {txt_path}")
        return

    os.makedirs(output_root, exist_ok=True)

    success = 0
    failed = 0

    # ⚠️ 用 utf-8-sig 自动处理 BOM
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        for line_num, line in enumerate(f, start=1):
            # 去除空白 + 引号
            src_path = line.strip().strip('"')

            if not src_path:
                continue

            # 统一路径格式（非常重要）
            src_path = os.path.normpath(src_path)

            if not os.path.isfile(src_path):
                print(f"⚠️ 第 {line_num} 行文件不存在: {src_path}")
                failed += 1
                continue

            filename = os.path.basename(src_path)
            name_without_ext, ext = os.path.splitext(filename)

            # 只处理 .h265
            if ext.lower() != ".h265":
                print(f"⚠️ 跳过非 h265 文件: {src_path}")
                continue

            dst_dir = os.path.join(output_root, name_without_ext)
            os.makedirs(dst_dir, exist_ok=True)

            dst_file = os.path.join(dst_dir, filename)

            try:
                shutil.copy2(src_path, dst_file)
                success += 1
                print(f"✅ 已复制: {filename}")
            except Exception as e:
                print(f"❌ 复制失败: {src_path}, 错误: {e}")
                failed += 1

    print("\n====== 处理完成 ======")
    print(f"成功: {success}")
    print(f"失败: {failed}")


if __name__ == "__main__":
    input_txt = r"C:\Users\chz62985\Desktop\liuyang\FT.txt"
    output_dir = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_70\input\liuyang\20251223"

    copy_files_from_txt(input_txt, output_dir)
