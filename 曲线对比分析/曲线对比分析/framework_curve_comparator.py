from GetSpecificAttributes import *
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class FrameworkCurveComparator:
    def __init__(self, video_path, recharge_path, version_old, version_new):
        self.video_path = video_path
        self.recharge_path = recharge_path
        self.version_old = version_old
        self.version_new = version_new

        self._video_filelist = None
        self._recharge_filelist = None

    def find_files(self, folder_path, extensions):
        """支持多后缀文件搜索"""
        filelist = {}
        for root, _, files in os.walk(folder_path):
            for file in files:
                for ext in extensions:
                    if file.endswith(ext):
                        name = os.path.splitext(file)[0]
                        if name not in filelist:
                            filelist[name] = os.path.join(root, file)
        return filelist

    def load_filelists(self):
        if self._video_filelist is None:
            self._video_filelist = self.find_files(
                self.video_path, extensions=[".h264", ".h265"]
            )

        if self._recharge_filelist is None:
            self._recharge_filelist = self.find_files(
                self.recharge_path, extensions=[".txt"]
            )

    def run_analysis(self, filename, start_frame, end_frame, target_id_old, target_id_new):
        self.load_filelists()

        self.filename = filename
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.target_id_old = target_id_old
        self.target_id_new = target_id_new

        # ===== 视频文件 =====
        videoapth = self._video_filelist.get(filename)
        if videoapth is None:
            raise FileNotFoundError(f"❌ 未找到视频文件: {filename} (.h264/.h265)")

        # ===== obj 日志 =====
        obj_txt = os.path.join(
            os.path.dirname(videoapth),
            "log",
            filename + "_arcsoft_obj.txt"
        )
        if not os.path.exists(obj_txt):
            raise FileNotFoundError(f"❌ 未找到 obj 日志文件: {obj_txt}")
        
        # ===== obj 日志 =====
        signal_txt = os.path.join(
            os.path.dirname(videoapth),
            "log",
            filename + "_Signal.txt"
        )
        if not os.path.exists(signal_txt):
            raise FileNotFoundError(f"❌ 未找到 signal_txt日志文件: {obj_txt}")

        # ===== recharge txt =====
        recharge_txt = self._recharge_filelist.get(filename)
        if recharge_txt is None:
            raise FileNotFoundError(f"❌ 未找到 recharge txt: {filename}.txt")

        arc_objdata_old = extract_objdata_arc(obj_txt)
        arc_objdata_new = extract_objdata_arc(recharge_txt)
        Ego_data = extract_ego_data(signal_txt)

        self.compare_curves(arc_objdata_old, arc_objdata_new, Ego_data)

    def compare_curves(self, arc_objdata_old, arc_objdata_new, Ego_data):
        frame_list = list(range(self.start_frame, self.end_frame))

        plt_list = [
            # "LongDistance",
            # "LatDistance",
            # "heading",
            # "AbsoluteLatVelocity",
            "AbsoluteLongVelocity",
            "AbsoluteLatVelocity",
            "RelativeLongVelocity",
            "RelativeLatVelocity",
            "f32Speed",
            "f32AbsoluteLongAcc"
        ]

        fig, axes = plt.subplots(3, 2, figsize=(9, 9))
        axes = axes.flatten()

        for idx, key in enumerate(plt_list):
            value_list_old = []
            value_list_new = []
            ego_speed_list = []

            for i in frame_list:
                try:
                    value = arc_objdata_old[i][self.target_id_old].get(key)
                    value_list_old.append(value if value is not None else np.nan)
                except Exception:
                    value_list_old.append(np.nan)

            for i in frame_list:
                try:
                    value = arc_objdata_new[i][self.target_id_new].get(key)
                    value_list_new.append(value if value is not None else np.nan)
                except Exception:
                    value_list_new.append(np.nan)
            
            for i in frame_list:
                try:
                    value = Ego_data[i].get(key)
                    ego_speed_list.append(value if value is not None else np.nan)
                except Exception:
                    ego_speed_list.append(np.nan)

            self.plot_framework_curves(
                frame_list,
                value_list_old,
                value_list_new,
                ego_speed_list,
                key,
                axes[idx]
            )

        plt.tight_layout()
        plt.show()

    def plot_framework_curves(self, frame_list, data_old, data_new, ego_speed_list, ylabel, ax):
        data_old = np.array(data_old, dtype=float)
        data_new = np.array(data_new, dtype=float)
        ego_speed_list = np.array(ego_speed_list, dtype=float)
        frame_list = np.array(frame_list, dtype=int)

        def insert_nan_if_discontinuous(frames, values):
            x, y = [], []
            last = None
            for f, v in zip(frames, values):
                if np.isnan(v):
                    x.append(f)
                    y.append(np.nan)
                else:
                    if last is not None and f != last + 1:
                        x.append(f - 0.5)
                        y.append(np.nan)
                    x.append(f)
                    y.append(v)
                    last = f
            return np.array(x), np.array(y)

        x_old, y_old = insert_nan_if_discontinuous(frame_list, data_old)
        x_new, y_new = insert_nan_if_discontinuous(frame_list, data_new)
        x_ego, y_ego = insert_nan_if_discontinuous(frame_list, ego_speed_list)

        def diff_std(arr):
            arr = arr[~np.isnan(arr)]
            return np.std(np.diff(arr)) if len(arr) > 1 else np.nan

        std_old = diff_std(y_old)
        std_new = diff_std(y_new)
        std_ego = diff_std(y_ego)

        ax.plot(x_old, y_old, label=f"{self.version_old} (STD={std_old:.6f})")
        ax.plot(x_new, y_new, label=f"{self.version_new} (STD={std_new:.6f})")
        ax.plot(x_ego, y_ego, label=f"EgoSpeed (STD={std_ego:.6f})", linestyle="--")

        ax.set_xlabel("Frame ID")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True)


if __name__ == "__main__":
    video_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_80\input\lishun\20260108\lynkco"
    recharge_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\CHZ\Southlake\adas_perception_v3.1_SPC030_2m_80\output_test\lishun\20260108_V3.1_2M_3.1.27223.1426"

    analyzer = FrameworkCurveComparator(
        video_path,
        recharge_path,
        version_old="实车",
        version_new="VMP100"
    )

    while True:
        try:
            print("请输入参数(输入 q 退出)")
            user_input = input(
                "格式: filename start_frame end_frame target_id_old target_id_new\n"
            ).strip()

            if user_input.lower() == "q":
                break

            parts = user_input.split()
            if len(parts) != 5:
                print("❌ 参数数量不正确")
                continue

            filename, start_frame, end_frame, tid_old, tid_new = parts

            analyzer.run_analysis(
                filename,
                int(start_frame),
                int(end_frame),
                int(tid_old),
                int(tid_new)
            )

            print(f"✅ {filename} 分析完成")

        except Exception as e:
            print(f"❌ 出错: {e}")
