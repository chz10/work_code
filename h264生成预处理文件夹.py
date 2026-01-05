import shutil
from pathlib import Path

def move_files_to_named_folder(root_folder: str, suffix=".h264", dry_run=False):
    """
    将指定后缀的文件移动到同名文件夹，避免重复嵌套并限制路径过长。
    """
    root_path = Path(root_folder)

    for file_path in root_path.rglob(f"*{suffix}"):
        file_stem = file_path.stem

        # 判断是否已经在合适的目录中，避免重复嵌套
        parent_name = file_path.parent.name
        if parent_name.startswith(file_stem):
        # if parent_name.startswith(file_stem):
            # print(f"文件已在合适的文件夹中，跳过: {file_path}")
            continue

        target_folder = file_path.parent / file_stem
        target_path = target_folder / file_path.name

        # 再次避免路径过长（预检查）
        if len(str(target_folder)) > 200:
            print(f"路径过长，跳过创建: {target_folder}")
            continue
        
        if "xin_data_" in str(file_path):
            continue

        # 创建目录
        if not target_folder.exists():
            if not dry_run:
                target_folder.mkdir(parents=True)
            print(f"创建目录: {target_folder}")

        # 移动文件
        if not target_path.exists():
            try:
                if not dry_run:
                    shutil.move(str(file_path), str(target_path))
                print(f"移动 {file_path} → {target_path}")
            except Exception as e:
                print(f"移动失败: {file_path} → {target_path}，错误: {e}")
        else:
            print(f"目标已存在，跳过: {target_path}")


def move_dir_contents_up(dir_path: Path, dry_run=False):
    """将 dir_path 下所有内容移动到父目录，并删除空目录"""
    parent_dir = dir_path.parent

    for item in dir_path.iterdir():
        dst_path = parent_dir / item.name

        if dst_path.exists():
            print(f"冲突文件，跳过: {dst_path}")
        else:
            if not dry_run:
                shutil.move(str(item), str(dst_path))
            print(f"移动 {item} → {dst_path}")

    try:
        if not dry_run:
            dir_path.rmdir()
        print(f"已删除空目录: {dir_path}")
    except OSError:
        print(f"目录非空或无法删除: {dir_path}")


def process_nested_dirs(root_folder: str, dry_run=False):
    """递归处理目录，将重复嵌套的文件夹内容上移"""
    for dir_path in sorted(Path(root_folder).rglob("*"), reverse=True):
        if dir_path.is_dir():
            parent = dir_path.parent
            if parent.name == dir_path.name:  # 嵌套检测
                print(f"检测到嵌套: {dir_path}")
                move_dir_contents_up(dir_path, dry_run=dry_run)


if __name__ == "__main__":
    folder_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\adas_perception_v3.1_SPC030_2m\input"
    # folder_path = input("请输入要处理的文件夹路径: ").strip()
    move_files_to_named_folder(folder_path, dry_run=False) # dry_run=True 表示仅预览不执行
    process_nested_dirs(folder_path, dry_run=False) # dry_run=True 表示仅预览不执行
