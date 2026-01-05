import os
import shutil

txt_path = r"C:\Users\chz62985\Desktop\素材路径.txt"
video_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m\input\20251124\AEB\2M\2M_20251017"
output_dir = r"C:\Users\chz62985\Desktop\素材"

os.makedirs(output_dir, exist_ok=True)

# 读取 txt 文件名
with open(txt_path, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

print("=== TXT 文件中读取到的文件名 ===")
for name in filenames:
    print(f"'{name}'  (长度: {len(name)})")
print("====================================\n")

found_count = 0

for root, dirs, files in os.walk(video_path):
    print(f"正在遍历目录: {root}")

    for file in files:
        # 打印实际文件名
        print(f"发现文件: '{file}' (长度: {len(file)})")

        if file in filenames:
            found_count += 1
            src = os.path.join(root, file)
            dst = os.path.join(output_dir, file)
            print(f"匹配成功！复制: {src}")
            shutil.copy2(src, dst)

print(f"\n匹配并复制完成，共找到 {found_count}/{len(filenames)} 个文件") # 修改为实际找到的文件数量