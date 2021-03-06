import argparse
import os
import pprint
import shutil
import sys
from utils.utils import get_all_about_data
from models.models import ft_net,PCB
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from tensorboardX import SummaryWriter

# import _init_paths
# import models
from config import config
from config import update_config
from function import train,validate,evaluate,get_id
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer,save_checkpoint,create_logger
# from utils.utils import save_checkpoint
# from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    #identify a model
    # model = eval('models.' + config.MODEL.NAME + '.get_cls_net')(
    #     config)
    if config.MODEL.PCB:
        model = PCB(751)
        print('you are using pcb')
    else:
        model = ft_net(751)
        print('you are using resnet')
    #can be deleted,i guess it shows the structure of model
    dump_input = torch.rand(
        (2, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    print('dump input ',type(dump_input))
    logger.info(get_model_summary(model, dump_input))
    print('get model summary above')

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    # [fuck] ,I can not understand this code
    # if os.path.exists(models_dst_dir):
    #     shutil.rmtree(models_dst_dir)
    # shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    #[tensorboard part]
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # even you use only one gpu.you also need to parallel, or when your multi-gpu model
    # can not be loaded in the future
    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])
            model.module.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch - 1
        )

    # Data loading code
    print(config.DATASET.ROOT)
    train_loader, valid_loader, \
    query_loader, query_cam, query_label, gallery_loader, gallery_cam, gallery_label \
        = get_all_about_data(config)


    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        lr_scheduler.step()
        # perf_indicator = validate(config, valid_loader, model, criterion,
        #                           final_output_dir, tb_log_dir, writer_dict)
        perf_indicator = evaluate(config,model,query_loader,gallery_loader,query_cam,query_label,gallery_cam,gallery_label)
        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        #if the code is terminted , this saved checkpoints can work
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            # 'state_dict': model.state_dict(),
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')


        # evaluate on validation set
        # perf_indicator = validate(config, valid_loader, model, criterion,
        #                           final_output_dir, tb_log_dir, writer_dict)
        # if epoch%config.TEST.TEST_GAP == 0:
        #     perf_indicator = evaluate(model,query_loader,gallery_loader,query_cam,
        #                               query_label,gallery_cam,gallery_label)
        #
        #     if perf_indicator > best_perf:
        #         best_perf = perf_indicator
        #         best_model = True
        #     else:
        #         best_model = False
        #
        #     logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        #     #if the code is terminted , this saved checkpoints can work
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'model': config.MODEL.NAME,
        #         # 'state_dict': model.state_dict(),
        #         'state_dict': model.module.state_dict(),
        #         'perf': perf_indicator,
        #         'optimizer': optimizer.state_dict(),
        #     }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    # torch.save(model.state_dict(), final_model_state_file)
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
