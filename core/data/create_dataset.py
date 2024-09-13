import os
import imp
import time

import numpy as np
import torch

from core.utils.file_util import list_files
from configs import cfg
from .dataset_args import DatasetArgs


def _query_dataset(data_type):
    module = cfg[data_type].dataset_module
    module_path = module.replace(".", "/") + ".py"
    dataset = imp.load_source(module, module_path).Dataset
    return dataset


def _get_total_train_imgs(dataset_path):
    train_img_paths = \
        list_files(os.path.join(dataset_path, 'images'),
                                exts=['.png'])
    return len(train_img_paths)


def create_dataset(data_type='train', dataset_mode='ins_level'):
    if dataset_mode=='ins_level':
        dataset_name = cfg[data_type].dataset
        args = DatasetArgs.get(dataset_name)
        # customize dataset arguments according to dataset type
        args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor
        if data_type == 'progress':
            total_train_imgs = _get_total_train_imgs(args['dataset_path'])
            args['skip'] = total_train_imgs // 16
            args['maxframes'] = 16
        if data_type in ['eval_novel_view' or 'eval_novel_pose']:
            args['skip'] = cfg.render_skip
    
        dataset = _query_dataset(data_type)
        dataset = dataset(**args)

    elif dataset_mode=='cat_level':

        dataset_list = []
        for ins_idx, dataset_name in enumerate(cfg[data_type].dataset):
            args = DatasetArgs.get(dataset_name)
            # customize dataset arguments according to dataset type
            args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor
            if data_type == 'progress':
                args['maxframes'] = 4
                args['ray_shoot_mode'] = 'image'
                args['skip'] = 1
            if data_type in ['eval_novel_view', 'eval_pose_transfer']:
                args['skip'] = cfg.render_skip
            dataset_list.append(args)
    
        dataset = _query_dataset(data_type)
        dataset = dataset(dataset_list)

    return dataset


def _worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def create_dataloader(data_type='train', dataset_mode='ins_level'):
    cfg_node = cfg[data_type]

    batch_size = cfg_node.batch_size
    shuffle = cfg_node.shuffle
    drop_last = cfg_node.drop_last

    dataset = create_dataset(data_type=data_type, dataset_mode=dataset_mode)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=cfg.num_workers,
                                              worker_init_fn=_worker_init_fn)

    return data_loader
