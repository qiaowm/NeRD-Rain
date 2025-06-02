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
            img = Image.open(img_path)
            # 转换为PNG格式
            if img.format != 'PNG':
                img = img.convert('RGB')  # 确保兼容性
            imgs.append(img)
        else:
            imgs.append(Image.new('RGB', (256, 256), (255, 255, 255)))  # 占位空白图

    # 以第一张图片的尺寸为基准
    base_width, base_height = imgs[0].size

    # 调整所有图片的尺寸
    resized_imgs = [img.resize((base_width, base_height), Image.Resampling.LANCZOS) for img in imgs]

    # 拼接：一列，3x1
    new_img = Image.new('RGB', (base_width, base_height * 3))
    new_img.paste(resized_imgs[0], (0, 0))
    new_img.paste(resized_imgs[1], (0, base_height))
    new_img.paste(resized_imgs[2], (0, base_height * 2))

    # 保存为PNG格式
    output_filename = os.path.splitext(filename)[0] + '.png'
    new_img.save(os.path.join(output_dir, output_filename), format='PNG')