import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips  # 即使不使用，代码中已经导入，可以暂时保留

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device
# lpips_fn = lpips.LPIPS(net='alex').to(device) # 可以注释掉，因为不再使用

sde = util.DenoisingSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], device=device)
sde.set_model(model.model)
print(sum(p.numel() for p in model.model.parameters()))

degrad_sigma = opt["degradation"]["sigma"] # 这个变量在后续没有直接使用，可以考虑移除或者保留

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"] # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)
 
    test_times = []

    for ii, test_data in enumerate(test_loader):

        print(test_loader.dataset.opt["dataroot_GT"])
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["LQ_path"][0] # 直接使用LQ的路径
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ = test_data["LQ"] # 直接加载LQ数据

        model.feed_data(LQ, None) # 将GT设置为None
        tic = time.time()
        model.test(sde, sigma=degrad_sigma, save_states=True)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        output = util.tensor2img(visuals["Output"].squeeze())  # uint8
        LQ = util.tensor2img(visuals["Input"].squeeze())  # uint8

        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".tif")
        util.save_img(output, save_img_path)
        print(save_img_path)
        LQ_img_path = os.path.join(dataset_dir, img_name + "_noisy.png")
        util.save_img(LQ, LQ_img_path)

        logger.info(img_name) # 仅记录图像名称

        # break

    print(f"average test time: {np.mean(test_times):.4f}")