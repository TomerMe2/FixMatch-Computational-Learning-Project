# import yaml
# import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict

from omegaconf import DictConfig, OmegaConf
import hydra
from sklearn.model_selection import KFold

from datasets import *
from models import *
from experiments import *
from utils.utils import setup_default_logging


OUTER_KFOLD = 10
INNER_KFOLD = 3
N_SEARCHES = 50


@hydra.main(config_path='./config', config_name='config')
def main(CONFIG: DictConfig) -> None:
    # # configuration
    # parser = argparse.ArgumentParser(description='Generic runner for FixMatch')
    # parser.add_argument('--config',  '-c',
    #                     dest="filename",
    #                     metavar='FILE',
    #                     help =  'path to the config file',
    #                     default='config/config.yaml')

    # args = parser.parse_args()
    # with open(args.filename, 'r') as file:
    #     try:
    #         config_file = yaml.safe_load(file)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # CONFIG = edict(config_file)
    # print('==> CONFIG is: \n', OmegaConf.to_yaml(CONFIG), '\n')

    # initial logging file
    logger = setup_default_logging(CONFIG, string='Train')
    logger.info(CONFIG)

    # # For reproducibility, set random seed
    if CONFIG.Logging.seed == 'None':
        CONFIG.Logging.seed = random.randint(1, 10000)
    random.seed(CONFIG.Logging.seed)
    np.random.seed(CONFIG.Logging.seed)
    torch.manual_seed(CONFIG.Logging.seed)
    torch.cuda.manual_seed_all(CONFIG.Logging.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # get datasets
    data = LOADDATA[CONFIG.DATASET.loading_data](CONFIG.DATASET)

    cta = data.get_cta() if CONFIG.DATASET.strongaugment == 'CTA' else None

    dataset, test_dataset = data.get_vanila_dataset()
    dataset.transform = None
    test_dataset.transform = None

    best_threshold, best_lambda = None, None
    idx_outer_fold = 0
    best_model_outer, best_model_top1_acc_outer = None, None
    for outer_train, outer_val in KFold(n_splits=OUTER_KFOLD).split(range(len(dataset))):
        outer_fold_train_dataset, outer_fold_val_dataset = data.split_to_idxs(outer_train, outer_val, dataset)
        outer_fold_train_dataset.transform = None
        outer_fold_val_dataset.transform = None


        best_model_inner, best_model_top1_acc_inner = None, None
        for search_idx in range(N_SEARCHES):
            # optimization for:
            # what thershold we need to have a confident label
            CONFIG.EXPERIMENT.threshold = np.random.uniform(0.5, 0.99)
            # what loss is added to the confident label (1 is default)
            CONFIG.EXPERIMENT.lambda_unlabeled = np.random.uniform(0.5, 1)

            idx_inner_fold = 0
            best_model_random, best_model_top1_acc_random = None, None
            for inner_train, inner_val in KFold(n_splits=INNER_KFOLD).split(range(len(dataset))):
                inner_fold_train_dataset, inner_fold_val_dataset = data.split_to_idxs(inner_train,
                                                                                      inner_val, outer_fold_train_dataset)
                inner_fold_train_dataset.transform = None
                inner_fold_val_dataset.transform = None

                # build the simple CNN
                model = WRN_MODELS['SimpleColorCNN'](CONFIG.MODEL)

                # build wideresnet
                # model = WRN_MODELS[CONFIG.MODEL.name](CONFIG.MODEL)

                logger.info("[Model] Building model {}".format(CONFIG.MODEL.name))

                if CONFIG.EXPERIMENT.used_gpu:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device=device)

                experiment = EXPERIMENT[CONFIG.EXPERIMENT.name](
                    model, CONFIG.EXPERIMENT, cta)

                if cta:
                    labeled_training_dataset, unlabeled_training_dataset, valid_dataset, cta_dataset = data.vanilla_dataset_to_unlabeled(dataset)
                    experiment.cta_probe_loader(cta_dataset)
                else:
                    labeled_training_dataset, unlabeled_training_dataset, valid_dataset = data.vanilla_dataset_to_unlabeled(dataset)

                experiment.labelled_loader(labeled_training_dataset)
                if CONFIG.DATASET.loading_data != 'LOAD_ORIGINAL' and unlabeled_training_dataset != None:
                    experiment.unlabelled_loader(
                        unlabeled_training_dataset, CONFIG.DATASET.mu)
                experiment.validation_loader(valid_dataset)
                experiment.fitting()
                print("======= Training done =======")
                logger.info("======= Training done =======")
                experiment.test_loader(valid_dataset)
                test_loss, top1_acc, top5_acc = experiment.testing()
                print("======= Testing done =======")
                logger.info("======= Testing done =======")

                if best_model_random is None or top1_acc > best_model_top1_acc_random:
                    best_model_random = model
                    best_model_top1_acc_random = top1_acc
                    best_threshold = CONFIG.EXPERIMENT.threshold
                    best_lambda = CONFIG.EXPERIMENT.lambda_unlabeled

                idx_inner_fold += 1

            if best_model_inner is None or best_model_top1_acc_random > best_model_top1_acc_inner:
                best_model_inner = best_model_random
                best_model_top1_acc_inner = best_model_top1_acc_random

        experiment = EXPERIMENT[CONFIG.EXPERIMENT.name](
            best_model_inner, CONFIG.EXPERIMENT, cta)

        if cta:
            _, _, test_fold_dataset, cta_dataset = data.vanilla_dataset_to_unlabeled(
                dataset)
            experiment.cta_probe_loader(cta_dataset)
        else:
            _, _, test_fold_dataset = data.vanilla_dataset_to_unlabeled(
                dataset)

        experiment.test_loader(test_fold_dataset)
        print('======= Outer Fold Test ========')
        logger.info('======= Outer Fold Test ========')
        test_loss, top1_acc, top5_acc = experiment.testing()
        print(f'test loss: {test_loss}, top1 acc: {top1_acc}, top5 acc: {top5_acc}')
        logger.info(f'test loss: {test_loss}, top1 acc: {top1_acc}, top5 acc: {top5_acc}')

        idx_outer_fold += 1

        if best_model_outer is None or best_model_top1_acc_inner > best_model_top1_acc_outer:
            best_model_outer = best_model_inner
            best_model_top1_acc_outer = best_model_top1_acc_inner

    # TODO: COMPUTE METRICS ON THE BEST MODEL
    print(f'best threshold: {best_threshold}')
    print(f'best lambda: {best_lambda}')


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
