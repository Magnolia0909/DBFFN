import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import IuxrayMultiImageDataset


class DBDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(300),
                transforms.RandomCrop(256),
                transforms.RandomChoice([transforms.RandomHorizontalFlip(p=1),
                                         transforms.RandomRotation(45),
                                         transforms.RandomVerticalFlip(p=1), ]),
                transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
                transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=0, fillcolor=0),
                # transforms.GaussianBlur(5, sigma=(0.01, 1.9)),
                transforms.ToTensor(),
                transforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)

