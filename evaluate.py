import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(folder1, folder2):
    # 获取两个文件夹中的文件名
    files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    if len(files1) != len(files2):
        raise ValueError("两个文件夹中的图片数量不一致！")

    total_psnr = 0
    total_ssim = 0
    num_files = len(files1)

    for file1, file2 in zip(files1, files2):
        if file1 != file2:
            raise ValueError(f"文件名不匹配：{file1} 和 {file2}")

        # 加载图片
        img1 = np.array(Image.open(os.path.join(folder1, file1)).convert('RGB'))
        img2 = np.array(Image.open(os.path.join(folder2, file2)).convert('RGB'))

        # 转换为 YCbCr 并提取 Y 通道
        img1_y = rgb_to_ycbcr(img1)[:, :, 0]
        img2_y = rgb_to_ycbcr(img2)[:, :, 0]

        # 计算 PSNR 和 SSIM
        current_psnr = psnr(img1_y, img2_y, data_range=255)
        current_ssim = ssim(img1_y, img2_y, data_range=255)

        total_psnr += current_psnr
        total_ssim += current_ssim

    # 计算平均值
    avg_psnr = total_psnr / num_files
    avg_ssim = total_ssim / num_files

    return avg_psnr, avg_ssim

def rgb_to_ycbcr(img):
    """将 RGB 图像转换为 YCbCr 格式"""
    img = img.astype(np.float32)
    ycbcr = np.empty_like(img)
    ycbcr[..., 0] = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    ycbcr[..., 1] = -0.168736 * img[..., 0] - 0.331264 * img[..., 1] + 0.5 * img[..., 2] + 128
    ycbcr[..., 2] = 0.5 * img[..., 0] - 0.418688 * img[..., 1] - 0.081312 * img[..., 2] + 128
    return ycbcr

if __name__ == "__main__":
    folder1 = '/home/featurize/data/LHP-Rain-RGB/test/target_downsample'
    folder2 = '/home/featurize/data/LHP-Rain-RGB/test/result_downsample_ft/'

    try:
        avg_psnr, avg_ssim = calculate_metrics(folder1, folder2)
        print(f"平均 PSNR: {avg_psnr:.2f}")
        print(f"平均 SSIM: {avg_ssim:.4f}")
    except Exception as e:
        print(f"发生错误: {e}")