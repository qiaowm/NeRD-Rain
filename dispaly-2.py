import os
from PIL import Image

# 三个文件夹路径
folders = [
    './Datasets/real-ds/input',
    './Datasets/real-ds/result_pretrain',
    './Datasets/real-ds/result_ft'
]

# 输出文件夹
output_dir = 'output/real-ds'
os.makedirs(output_dir, exist_ok=True)

# 获取所有文件名（假设三个文件夹文件名完全一致）
filenames = sorted(os.listdir(folders[0]))

for filename in filenames:
    imgs = []
    for folder in folders:
        img_path = os.path.join(folder, filename)
        if os.path.exists(img_path):
            imgs.append(Image.open(img_path))
        else:
            imgs.append(Image.new('RGB', (256, 256), (255, 255, 255)))  # 占位空白图

    # 假设所有图片尺寸一致
    w, h = imgs[0].size

    # 拼接：一列，3x1
    new_img = Image.new('RGB', (w, h * 3))
    new_img.paste(imgs[0], (0, 0))
    new_img.paste(imgs[1], (0, h))
    new_img.paste(imgs[2], (0, h * 2))

    # 保存
    new_img.save(os.path.join(output_dir, filename))