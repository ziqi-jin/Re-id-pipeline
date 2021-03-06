import os
import logging
import time
from pathlib import Path

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from function import get_id
def create_logger(cfg, cfg_name, phase='train'):
    '''
    :param cfg:the config file include all the training parameters
    :param cfg_name: the name of cfg_name e.g. path/to/test.yaml,
    :param phase: it is a flag which will be used when create a log file and give it a name
    :return:
        logger :logger can record log files
        final_output_dir : the dir which save log file
        tensorboard_log_dir : the dir which save tesnsorboard log file
    '''
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    '''
    :param cfg: config file
    :param model: model you want to train
    :return: a optimizer which is required in cfg file
    '''
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        if not cfg.MODEL.PCB:
            ignored_params = list(map(id, model.module.classifier.parameters() ))
            base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
            optimizer = optim.SGD([
                    {'params': base_params, 'lr': 0.1*cfg.TRAIN.LR},
                    {'params': model.module.classifier.parameters(), 'lr': cfg.TRAIN.LR}
                ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, model.module.model.fc.parameters() ))
            ignored_params += (list(map(id, model.module.classifier0.parameters() )) 
                            +list(map(id, model.module.classifier1.parameters() ))
                            +list(map(id, model.module.classifier2.parameters() ))
                            +list(map(id, model.module.classifier3.parameters() ))
                            +list(map(id, model.module.classifier4.parameters() ))
                            +list(map(id, model.module.classifier5.parameters() ))
                            #+list(map(id, model.classifier6.parameters() ))
                            #+list(map(id, model.classifier7.parameters() ))
                            )
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = optim.SGD([
                    {'params': base_params, 'lr': 0.1*cfg.TRAIN.LR},
                    {'params': model.module.model.fc.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.module.classifier0.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.module.classifier1.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.module.classifier2.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.module.classifier3.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.module.classifier4.parameters(), 'lr': cfg.TRAIN.LR},
                    {'params': model.module.classifier5.parameters(), 'lr': cfg.TRAIN.LR},
                    #{'params': model.classifier6.parameters(), 'lr': 0.01},
                    #{'params': model.classifier7.parameters(), 'lr': 0.01}
                ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )
    return optimizer

def get_all_about_data(config):
    '''
    this function can give you all the dataloaders you will use
    :param config: config file
    :return: train ,valid ,query ,gallery dataloaders
    '''
    gpus = list(config.GPUS)
    traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)
    query_dir = os.path.join(config.DATASET.ROOT, config.DATASET.QUERY)
    gallery_dir = os.path.join(config.DATASET.ROOT, config.DATASET.GALLERY)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #################################################
    if config.MODEL.PCB:
        transform_train_list = [
            transforms.Resize((384,192), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        transform_val_list = [
            transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    else :
        transform_train_list = [
            # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((256, 128), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        transform_val_list = [
            transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    ###########################################
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(transform_train_list)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(transform_val_list)),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
    )
    query_dataset = datasets.ImageFolder(query_dir, transforms.Compose(transform_val_list))
    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
    )
    gallery_dataset = datasets.ImageFolder(gallery_dir, transforms.Compose(transform_val_list))
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
    )
    gallery_path = gallery_dataset.imgs
    query_path = query_dataset.imgs
    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)
    return train_loader,valid_loader,query_loader,query_cam,query_label,gallery_loader,gallery_cam,gallery_label

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))

