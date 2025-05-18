import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_image(input_path, output_path):
    """
    Convert a single image to RGB format and save it to the LHP path.
    """
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if not already in RGB mode
            rgb_img = img.convert("RGB")
            rgb_img.save(output_path)
            print(f"Converted: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def convert_images_to_rgb(input_dir, output_dir, max_workers=8):
    """
    Recursively convert all images in a directory to RGB format and save them to an LHP directory.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the LHP directory to save converted images.
        max_workers (int): Maximum number of threads to use.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(input_dir):
            for file in files:
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path, file)

                # Ensure the LHP subdirectory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Submit the image processing task to the thread pool
                tasks.append(executor.submit(process_image, input_path, output_path))

        # Wait for all tasks to complete
        for task in tasks:
            task.result()

# Example usage
input_directory = "/home/weiming/PycharmProjects/NeRD-Rain/Datasets/LHP-Rain/train/target"
output_directory = "/home/weiming/PycharmProjects/NeRD-Rain/Datasets/LHP-Rain-RGB/train/target"
convert_images_to_rgb(input_directory, output_directory, max_workers=12)