import cv2
import numpy as np
import os
import gc
from skimage.morphology import opening, closing, disk
from numpy.linalg import svd
import matplotlib.pyplot as plt


def histogram_equalization(image):
    """
    执行直方图均衡化以增强对比度
    """
    if len(image.shape) == 3:  # 彩色图像
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:  # 灰度图像
        return cv2.equalizeHist(image)


def guided_filter(image, radius=5, eps=0.1):
    """
    应用引导滤波器
    """
    radius = int(radius)  # 确保 radius 是整数
    if len(image.shape) == 3:  # 彩色图像
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            guided_filter = cv2.ximgproc.createGuidedFilter(guide=image[:, :, i].astype(np.float32),
                                                            radius=radius, eps=eps)
            result[:, :, i] = guided_filter.filter(src=image[:, :, i].astype(np.float32))
        return np.clip(result, 0, 255).astype(np.uint8)
    else:  # 灰度图像
        guided_filter = cv2.ximgproc.createGuidedFilter(guide=image.astype(np.float32),
                                                        radius=radius, eps=eps)
        result = guided_filter.filter(src=image.astype(np.float32))
        return np.clip(result, 0, 255).astype(np.uint8)

def detect_rain_lines(image, ksize=15, threshold=20):
    """
    使用方向滤波器和形态学操作检测雨线
    """
    # 转为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 计算垂直方向的梯度
    kernel_vertical = np.ones((ksize, 1), np.float32) / ksize
    vertical_gradient = cv2.filter2D(gray, -1, kernel_vertical)

    # 应用阈值处理突出雨线
    _, rain_mask = cv2.threshold(vertical_gradient, threshold, 255, cv2.THRESH_BINARY)

    # 应用形态学操作增强雨线
    rain_mask = closing(rain_mask, disk(1))
    rain_mask = opening(rain_mask, disk(1))

    return rain_mask


def low_rank_decomposition(image, mask, lambd=0.02, iterations=10):
    """
    使用低秩矩阵分解分离雨线和背景
    简化版本的低秩分解，使用SVD实现
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        background = np.zeros_like(image, dtype=np.float32)
        rain = np.zeros_like(image, dtype=np.float32)

        for i in range(c):
            channel = image[:, :, i].astype(np.float32)
            bg, r = low_rank_channel(channel, mask, lambd, iterations)
            background[:, :, i] = bg
            rain[:, :, i] = r
    else:
        background, rain = low_rank_channel(image.astype(np.float32), mask, lambd, iterations)

    return np.clip(background, 0, 255).astype(np.uint8), np.clip(rain, 0, 255).astype(np.uint8)


def low_rank_channel(channel, mask, lambd, iterations):
    """
    对单通道应用低秩分解
    """
    # 初始化背景和雨线
    background = channel.copy()
    rain = np.zeros_like(channel)

    for _ in range(iterations):
        # 更新背景 (低秩部分)
        U, S, Vt = svd(background, full_matrices=False)
        # 软阈值处理
        S_threshold = np.maximum(S - lambd, 0)
        background = U @ np.diag(S_threshold) @ Vt

        # 更新雨线 (稀疏部分)
        rain = channel - background
        # 使用掩码约束雨线区域
        rain = rain * (mask / 255.0)

        # 更新背景
        background = channel - rain

    return background, rain


def enhance_details(image, alpha=1.5, beta=0):
    """
    增强图像细节
    """
    return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)


def derain_image(image_path, save_path=None):
    """
    完整的去雨流程
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 步骤1: 预处理
    enhanced_image = histogram_equalization(image)
    filtered_image = guided_filter(enhanced_image, radius=3, eps=0.1)

    # 步骤2: 雨线检测
    rain_mask = detect_rain_lines(filtered_image, ksize=9, threshold=15)

    # 步骤3: 雨线分离
    background, rain = low_rank_decomposition(filtered_image, rain_mask, lambd=0.01, iterations=5)

    # 步骤4: 图像重建与增强
    final_result = enhance_details(background, alpha=1.2)
    brightness_diff = np.mean(image) - np.mean(final_result)
    final_result = np.clip(final_result + brightness_diff, 0, 255).astype(np.uint8)

    # 步骤5: 后处理
    final_result = guided_filter(final_result, radius=2, eps=0.05)

    # 保存结果
    if save_path:
        cv2.imwrite(save_path, final_result)
        print(f"结果已保存至: {save_path}")

    return final_result
def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = sorted(os.listdir(input_dir))

    for file_name in file_list:
        input_path = os.path.join(input_dir, file_name)
        if os.path.isfile(input_path) and file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            output_path = os.path.join(output_dir, file_name)
            try:
                # 逐个处理图像
                derain_image(input_path, output_path)
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")
            finally:
                # 释放内存
                gc.collect()

if __name__ == "__main__":
    input_dir = './Datasets/LHP-Rain-RGB/result/result_ft'
    output_dir = './Datasets/LHP-Rain-RGB/result/result_ft_postprocess'
    process_folder(input_dir, output_dir)