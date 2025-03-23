import os
import imageio
import os.path as osp
import concurrent.futures  # 导入 concurrent.futures 模块

def process_image_task(image_path, problematic_images):
    """
    任务函数：尝试读取单个图片文件，如果失败则将路径添加到 problematic_images 列表。
    （注意函数名已更改为 process_image_task）

    Args:
        image_path: 图片文件路径
        problematic_images: 共享的问题图片列表
    """
    try:
        imageio.imread(image_path)  # 尝试读取图片
        print(image_path)
    except Exception as e:
        print(f"线程 {threading.current_thread().name} 读取图片失败: {image_path}") # 仍然使用 threading.current_thread().name 获取线程名
        print(f"线程 {threading.current_thread().name} 错误信息: {e}")
        problematic_images.append(image_path)


def find_problematic_images_executor(image_dir, num_threads=4):
    """
    使用 ThreadPoolExecutor 多线程扫描指定图片文件夹，找出无法读取的图片文件。
    （注意函数名已更改为 find_problematic_images_executor）

    Args:
        image_dir: 图片文件夹路径
        num_threads: 使用的线程数量，默认为 4
    """
    problematic_images = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_paths = [] # 先收集所有图片路径

    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(osp.join(image_dir, filename)) # 将图片路径添加到列表中

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor: # 使用 ThreadPoolExecutor 管理线程池
        futures = [executor.submit(process_image_task, path, problematic_images) for path in image_paths] # 提交任务到线程池

        concurrent.futures.wait(futures) # 等待所有任务完成 (替代 thread.join() 循环)


    if problematic_images:
        print("\n--- 发现以下问题图片 ---")
        for path in problematic_images:
            print(path)
    else:
        print(f"在 {image_dir} 文件夹下未发现读取错误的图片。")


if __name__ == "__main__":
    import threading # 需要导入 threading 才能在 process_image_task 中使用 threading.current_thread().name
    target_dir = 'dataset/LSDIR/train'
    num_threads_to_use = 12 # 再次尝试较小的线程数，例如 2
    if not osp.exists(target_dir):
        print(f"错误: 文件夹路径 '{target_dir}' 不存在，请检查路径是否正确。")
    else:
        find_problematic_images_executor(target_dir, num_threads=num_threads_to_use) # 调用新的函数名