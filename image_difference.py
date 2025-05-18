from PIL import Image, ImageChops

def image_difference(image1_path, image2_path, output_path):
    # 打开两张图像
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 确保两张图像的尺寸一致
    if img1.size != img2.size:
        raise ValueError("两张图像的尺寸不一致，无法作差")

    # 计算图像差异
    diff = ImageChops.difference(img1, img2)

    # 保存结果图像
    diff.save(output_path)
    print(f"图像差异已保存到: {output_path}")

def subtract_difference(image1_path, difference_path, output_path):
    # 打开第一张图像和差异图像
    img1 = Image.open(image1_path)
    diff = Image.open(difference_path)

    # 确保两张图像的尺寸一致
    if img1.size != diff.size:
        raise ValueError("图像尺寸不一致，无法进行减法操作")

    # 从第一张图像中减去差异图像
    result = ImageChops.subtract(img1, diff)

    # 保存结果图像
    result.save(output_path)
    print(f"结果图像已保存到: {output_path}")

# 示例用法
image1 = '/home/weiming/PycharmProjects/NeRD-Rain/results/LHP-Rain/2701.png'  # 第一张图像路径
image2 = '/home/weiming/PycharmProjects/NeRD-Rain/Datasets/LHP-Rain/test/input_downsample/2701.png'  # 第二张图像路径
output = './difference.png'  # 差异图像保存路径

image_difference(image1, image2, output)
# 减去差异图像
output_subtract = './result.png'  # 减法结果图像保存路径
subtract_difference(image1, output, output_subtract)