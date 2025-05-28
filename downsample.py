import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def process_image(filename, input_dir, output_dir):
    input_path = os.path.join(input_dir, filename)

    # 检查是否为图片文件
    if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        with Image.open(input_path) as img:
            # 获取原始尺寸
            width, height = img.size
            # 计算新尺寸（四分之一大小）
            new_size = (width // 2, height // 2)
            # 调整图片大小
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            # 保存到输出文件夹
            output_path = os.path.join(output_dir, filename)
            img_resized.save(output_path)


def downsample_images(input_dir, output_dir):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入文件夹中的所有文件
    filenames = os.listdir(input_dir)

    # 使用线程池并行处理图片
    with ThreadPoolExecutor() as executor:
        executor.map(lambda filename: process_image(filename, input_dir, output_dir), filenames)


# 示例用法
input_folder = './Datasets/real'  # 输入文件夹路径
output_folder = './Datasets/real-ds/input'  # 输出文件夹路径
downsample_images(input_folder, output_folder)