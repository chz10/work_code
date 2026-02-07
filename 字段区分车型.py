import os
import json
import shutil


# =========================
# 车型判断
# =========================
def judge_vehicle(front, rear):
    if front == 940 and rear == 1105:
        return "geely"
    if front == 906 and rear == 1070:
        return "lynkco"
    if front == 1610 and rear == 1620:
        return "wuling_5577"
    if front == 963 and rear == 974:
        return "wuling_5741"
    return "unknown"


# =========================
# 读取 calibration.json
# =========================
def read_calibration_fields(calib_path):
    for enc in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            with open(calib_path, "r", encoding=enc) as f:
                data = json.load(f)
            info = data.get("vehicleInfo", {})
            return (
                info.get("s32XDistanceFromFront"),
                info.get("s32XDistanceFromRear"),
            )
        except Exception:
            continue
    raise ValueError("JSON 解析失败")


# =========================
# 扫描并建立【文件夹 → 车型】映射
# =========================
def analyze_folders(root_dir):
    mapping = {}  # folder -> vehicle

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith("_calibration.json"):
                continue

            calib_path = os.path.join(root, file)

            # VisInsight_xxx 目录（log 的上一级）
            vis_folder = os.path.abspath(os.path.join(root, ".."))

            try:
                front, rear = read_calibration_fields(calib_path)
                vehicle = judge_vehicle(front, rear)
            except Exception:
                vehicle = "unknown"

            mapping[vis_folder] = vehicle

    return mapping


# =========================
# Dry-run 预览
# =========================
def preview(mapping, root_dir):
    print("\n📋【复制预览（不会修改任何源文件）】\n")

    for folder, vehicle in mapping.items():
        target = os.path.join(root_dir, vehicle, os.path.basename(folder))
        print(f"📂 源目录: {folder}")
        print(f"➡️  将复制到: {target}\n")

    print("=" * 60)
    print(f"📦 预计复制文件夹数量: {len(mapping)}")


# =========================
# 真正执行复制
# =========================
def execute(mapping, root_dir):
    for folder, vehicle in mapping.items():
        target_root = os.path.join(root_dir, vehicle)
        os.makedirs(target_root, exist_ok=True)

        target = os.path.join(target_root, os.path.basename(folder))

        if os.path.exists(target):
            print(f"⚠️ 目标已存在，跳过复制: {target}")
            continue

        shutil.copytree(folder, target)
        print(f"✅ 已复制: {folder} → {target}")


# =========================
# 主流程
# =========================
def main():
    root_dir = r"\\GZY72677-2350\Badcase\FTvideo\badcase_2m"

    mapping = analyze_folders(root_dir)

    preview(mapping, root_dir)

    answer = input("\n❓ 是否确认执行复制？请输入 YES 执行，其它任意键退出：")

    if answer == "YES":
        print("\n📦 开始执行复制...\n")
        execute(mapping, root_dir)
        print("\n✅ 复制完成（源数据未做任何修改）")
    else:
        print("\n🛑 已取消，未进行任何复制操作")


if __name__ == "__main__":
    main()
