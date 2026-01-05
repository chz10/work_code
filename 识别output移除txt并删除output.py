import os
import shutil

def process_version_date_dir(version_date_dir):
    if not os.path.isdir(version_date_dir):
        print(f"❌ 路径不存在: {version_date_dir}")
        return

    for car_type in os.listdir(version_date_dir):
        car_dir = os.path.join(version_date_dir, car_type)
        if not os.path.isdir(car_dir):
            continue

        output_dir = os.path.join(car_dir, "output")
        if not os.path.isdir(output_dir):
            continue

        print(f"🚗 处理车型: {car_type}")

        moved = 0
        for file in os.listdir(output_dir):
            if not file.lower().endswith(".txt"):
                continue

            src = os.path.join(output_dir, file)
            dst = os.path.join(car_dir, file)

            if os.path.exists(dst):
                base, ext = os.path.splitext(file)
                dst = os.path.join(car_dir, f"{base}_copy{ext}")

            shutil.move(src, dst)
            moved += 1

        if not os.listdir(output_dir):
            shutil.rmtree(output_dir)
            print("🗑️ output 已删除")

        print(f"✅ 移动 txt 数量: {moved}\n")


if __name__ == "__main__":
    version_date_dir = (
        r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake"
        r"\adas_perception_v3.1_SPC030_2m_60"
        r"\output\BrakeLightOn\20251219_test"
    )

    process_version_date_dir(version_date_dir)


#\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_60\output\BrakeLightOn\20251219_test\lixiang1