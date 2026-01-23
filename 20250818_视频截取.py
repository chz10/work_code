
import os
import subprocess


def format_time(seconds):
    """
    将秒数格式化为FFmpeg可读取的时间格式
    """
    hh = int(seconds // 3600)
    mm = int((seconds - hh * 3600) // 60)
    ss = int(seconds - hh * 3600 - mm * 60)
    ff = int((seconds - int(seconds)) * 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ff:03d}"

def get_new_h264(src_path, save_path, start_frame, end_frame):
    start_second = int(start_frame / 30)    # 按照固定30帧计算
    end_second = int(end_frame / 30)
    ffmpeg_path = r"C:\Users\chz62985\Desktop\ADAS_Visualization_3.6.9_20260109\ffmpeg_tool\ffmpeg.exe"
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.h265'):
                file_path = os.path.join(root, file)
                dir_name = file.replace('.h265', '') + f'_{start_frame}to{end_frame}'
                out_file_h264 = os.path.join(save_path, dir_name, dir_name + '.h265')
                os.makedirs(os.path.dirname(out_file_h264), exist_ok=True)
                print(out_file_h264)
                command = [
                    ffmpeg_path, "-loglevel", "quiet", "-y", "-i", file_path, "-vcodec", "copy", "-acodec", "copy",
                    "-ss", format_time(start_second), "-to", format_time(end_second), out_file_h264
                ]
                subprocess.run(command, check=False)


if __name__ == '__main__':
    # 截取视频片段，视频所在路径、保存路径、开始帧、结束帧
    src_path = r"\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\DWZ\test_data\FT\JIRA\VisInsight_20260109140457"
    save_path = r"C:\Users\chz62985\Desktop\dwz"
    start_frame = 4800
    end_frame = 5510

    get_new_h264(src_path, save_path, start_frame, end_frame)