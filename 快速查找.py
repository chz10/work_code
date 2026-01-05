import os
import multiprocessing as mp
from tqdm import tqdm


def search_worker(args):
    root_path, keyword = args
    for dirpath, dirnames, filenames in os.walk(root_path):
        for d in dirnames:  # 搜索文件夹
            if keyword in d.lower():
                return os.path.join(dirpath, d)

        for f in filenames:  # 搜索文件
            if keyword in f.lower():
                return os.path.join(dirpath, f)

    return None


def find_file_super_fast(root_path, keyword, process_num=6):
    keyword = keyword.lower()  # 不区分大小写

    # 先获取一级目录
    subdirs = [os.path.join(root_path, d)
               for d in os.listdir(root_path)
               if os.path.isdir(os.path.join(root_path, d))]

    if not subdirs:
        subdirs = [root_path]

    # 限制进程数
    process_num = min(process_num, len(subdirs))

    tasks = [(d, keyword) for d in subdirs]

    result = None
    with mp.Pool(process_num) as pool:
        # tqdm 进度条
        for res in tqdm(pool.imap_unordered(search_worker, tasks),
                        total=len(tasks),
                        desc="搜索进度",
                        unit="目录"):
            if res:
                result = res
                pool.terminate()  # 立即停止其它进程
                break

    return result



if __name__ == "__main__":
    root = "\\\\Material\\xuekangkang\\download\\1009224\\"  # 根目录
    keyword = "20250913194101"          # 模糊关键字

    result = find_file_super_fast(root, keyword)
    if result:
        print("找到：", result)
    else:
        print("未找到匹配的文件或文件夹")

