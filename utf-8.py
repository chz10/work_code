import os

root = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_8m\output\duwenzhe\20260202_V3.1_8M_3.1.27223.1525\geely\output\VisInsight_20260113111847.txt"

for root_dir, _, files in os.walk(root):
    for name in files:
        if name.endswith(".txt"):
            path = os.path.join(root_dir, name)
            try:
                with open(path, "r", encoding="gbk") as f:
                    content = f.read()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("✅ 转换成功:", path)
            except Exception as e:
                print("❌ 失败:", path, e)
