import argparse
import torchvision.models as models
import datetime


def parse_args():


    parser = argparse.ArgumentParser(description='VAE Training')
    parser.add_argument('--data-dgx', metavar='DIR',default='/raid/vae',
                        help='path to dataset')
    parser.add_argument('--data-summitdev', metavar='DIR',default='',
                        help='path to dataset on summitdev')
    parser.add_argument('--data-summit', metavar='DIR',default='',
                        help='path to dataset')
    parser.add_argument('--data-local', metavar='DIR',default='/home/fa6/data/deidentified/',
                        help='path to dataset')
    parser.add_argument('--env',default='local',type=str,
                        help='select execution environment (default: summitdev)')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='S',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-train', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('--batch-val', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 40)')
    parser.add_argument('--batch-test', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 40)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate (default 1e-4)')
    parser.add_argument('--step-size', default=10, type=int,
                        metavar='SS', help='learning rate scheduler step size')
    parser.add_argument('--gamma', default=.1, type=float,
                        metavar='G', help='learning rate scheduler gamma')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPTIM',
                        help='type of optimizer (default=Adam)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 21)')
    parser.add_argument('--resume', default='checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: checkpoint.pth.tar)')
    parser.add_argument('--best-model', default='model.pth.tar', type=str, metavar='PATH',
                        help='save best model (default: model.pth.tar)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-t','--tag',default='trial_{x:%Y%m%d%H%M%S}'.format(x=datetime.datetime.now()),type=str,
                        help='unique tag for storing experiment data')
    parser.add_argument('--data-store',default='data/trial_{x:%Y%m%d%H%M%S}',type=str,
                        help='unique tag for storing experiment data')
    parser.add_argument('-f','--factor',default=0.1,type=float,
                        metavar='F',help='learning rate sheduler factor (default: .1)')
    parser.add_argument('--patience',default=5,type=int,
                        metavar='P',help='learning rate sheduler patience (default: 5)')
    parser.add_argument('--trial', dest='trial', action='store_true',
                        help='trial experiment with smaller dataset (default: False)')
    parser.add_argument('-v', '--version', dest='version', type=float, default=0.4,
                        help='torch version (default: 0.4)')
    parser.add_argument('--stop-patience', default=10, type=int,
                        help='patience before early termination of training (default: 10)')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for experiments. [default: 42]')
    parser.add_argument('--fp16', action='store_true',
                        help='running model in half precision')
    parser.add_argument('--parallel', action='store_true',
                        help='running model in parallel (default: False)')
    parser.add_argument('--distributed', action='store_true',
                        help='running model in distributed parallel (default: False)')
    parser.add_argument('--n-nodes', default=1, type=int,
                        help='number of nodes (default: 14)')
    parser.add_argument('--n-threads', default=4, type=int,
                        help='number of threads (default: 14)')
    parser.add_argument('--n-gpus', default=4, type=int,
                        help='number of gpus (default: 4)')
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        help='number of warmup epochs (default: 5)')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')

    parser.add_argument('--augment', action='store_true',
                        help='enable torch augment (default: False)')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend (default: nccl')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training (default: \'env://\'')

    parser.add_argument('--drop-rate', default=.3, type=float,
                        help='drop out rate (default: .3)')

    args = parser.parse_args()
    # print(args.block_config)
    return args



def local_args():


    class argtuple():

        __args = {'arch': 'densenet121', 'augment': False, 'base_lr': 0.0125, 'batch_test': 5, 'batch_train': 5,
                'best_model': 'model.pth.tar', 'crop': 0.875, 'data_dgx': '',
                'data_summit': '', 'batch_val': 5,
                'data_summitdev': '', 'env': 'local',
                'data_local':'/home/fa6/data/',
                'epochs': 300, 'evaluate': False, 'factor': 0.1, 'fp16': False, 'gamma': 0.1, 'hvd': True,
                'lr': 0.0001, 'momentum': 0.9, 'num_classes': 14, 'n_gpus': 4, 'n_nodes': 1, 'n_threads': 4,
                'no_cuda': False,'cuda':True,'distributed':False,
                'optimizer': 'Adam', 'parallel': False, 'patience': 5, 'pretrained': False, 'print_freq': 50,
                'resources': '244','hvd_size':1,'hvd_rank':0,'hvd_local_rank':1,
                'resume': 'checkpoint.pth.tar', 'seed': 42, 'start_epoch': 0, 'step_size': 30, 'stop_patience': 10,
                'tag': 'data/experiment',
                'train_list': './chestX-ray14/labels/train_list.txt', 'trial': False, 'weight_decay': 1e-05,
                'version': 0.4, 'warmup_epochs': 5, 'workers': 0}
        __args['data'] = {'summitdev': __args['data_summitdev'],
                               'summit': __args['data_summit'],
                               'dgx': __args['data_dgx'],
                               'local':__args['data_local']}[__args['env']]
        __args['batch'] = {'train': __args['batch_train'],
                               'test': __args['batch_test'],
                               'val': __args['batch_val']}

        def __init__(self, **kwargs):
            if kwargs:
                self.__args = kwargs

        # @property
        def __getattr__(self, key):
            if key == '__args':
                return self.__args
            # try:
            return self.__args[key]
            # except KeyError:
            #     print('{} arg does not exist!')
            #     return None


    return argtuple()
