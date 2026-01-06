import os

def get_files_list(path, txt_all, keywords):
    for root, dirs, files in os.walk(path):
        for file in files:
            
            if file.endswith(keywords):
                file_path = os.path.join(root, file)
                file_name = file.replace('.h264', '\n')
                if file_name in txt_all:
                    result_pd.write(file_path + '\n')


if __name__ == '__main__':
    src_video_path = r'\\hz-iotfs02\Model_Test\TestSpace\Personal_Space\GZY\IHBC'
    txt_file = r"C:\Users\chz62985\Desktop\gaoziyi\test_gzy.txt"

    result_pd = open('F:222.txt', 'w+', encoding='utf-8')

    txt_all = open(txt_file, 'r').readlines()

    get_files_list(src_video_path, txt_all, '.h264')