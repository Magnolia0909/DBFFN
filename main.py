import torch
import argparse
import numpy as np
from tokenizers import Tokenizer
from dataloaders import DBDataLoader
from metrics import compute_scores
from optimizers import build_optimizer, build_lr_scheduler
from trainer import Trainer
from loss import compute_loss
from dbffn import DBModel
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_agrs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='', help='the path to the directory containing the data.')
    parser.add_argument('--dataset_name', type=str, default='iu_xray', help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')

    # define of dual-branch
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')
    parser.add_argument('--visual_extractor_02', type=str, default='ViT', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_02_image_size', type=int, default=256, help='image size of visual extractor')
    parser.add_argument('--visual_extractor_02_patch_size', type=int, default=16, help='patch size of visual extractor')
    parser.add_argument('--visual_extractor_02_num_classes', type=int, default=1000,  help='num classes of visual extractor')
    parser.add_argument('--visual_extractor_02_dim', type=int, default=2048, help='dim of visual extractor')
    # PiT
    # parser.add_argument('--visual_extractor_02_depth', type=list, default=(3,3,3), help='depth of visual extractor')
    parser.add_argument('--visual_extractor_02_depth', type=int, default=3, help='depth of visual extractor')
    parser.add_argument('--visual_extractor_02_heads', type=int, default=16, help='head numbers of visual extractor')
    parser.add_argument('--visual_extractor_02_dropout', type=float, default=0.1, help='dropout of visual extractor')
    parser.add_argument('--visual_extractor_02_mlp_dim', type=int, default=2048, help='mlp dim of visual extractor')
    parser.add_argument('--visual_extractor_02_emd_dropout', type=float, default=0.1, help='emd dropout of visual extractor')

    # Transformer
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    # parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    # parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=50, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, default=None, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--log_dir', type=str, default='', help='save log.')
    parser.add_argument('--report_dir', type=str, default='', help='report generated from process.')

    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    tokenizer = Tokenizer(args)

    train_dataloader = DBDataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = DBDataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = DBDataLoader(args, tokenizer, split='test', shuffle=False)


    model = DBModel(args, tokenizer)
    criterion = compute_loss
    metrics = compute_scores
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
