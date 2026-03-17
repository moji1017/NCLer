import torch
from data.cifar import CIFAR10
from data.news import News
from data.geodataset import GeoDataset


def get_dataloader(args):

    # cifar10 10 / mnist 10 / zircon _ 3 / apatite _ 11 / cifar100 100
    if args.dataset == 'cifar10':
        args.input_channel = 3
        args.num_classes = 10
        args.output = 10
        args.hidden = 128
        args.input_features = 10
        args.network = 'CNN7'
        args.classes_name = ['Airplane', 'Car', 'Birds', 'Cat', 'Deer', 'Dogs', 'Frog', 'Horse', 'Ship', 'Truck']
        train_dataset = CIFAR10(root='./data',
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate,
                                outlier_rate=args.outlier_rate,
                                imbalance_coefficient=args.imbalance_coefficient,
                                alpha=args.alpha,
                                download=True
                                )
    elif args.dataset == 'news':
        args.input_channel = 1  # 1 / 3
        args.num_classes = 10
        args.output = args.num_classes  # 10 if args.dataset == 'cifar10_csv' else args.cifar100_select_label
        args.input_features = 300  # 100 / 300 / 768 同时控制当前数据集的特征大小
        args.hidden = args.hidden_dim[-1]
        args.network = 'MLPs'
        args.classes_name = [
            'atheism', 'graphics', 'ms_windows', 'pc_hardware', 'mac_hardware', 'windows_x', 'forsale', 'autos', 'motorcycles', 'baseball',
            'hockey',
            'crypt', 'electronics', 'sci.med', 'sci.space', 'christian', 'talk.guns', 'talk.mideast', 'talk.misc', 'talk.religion'
        ]
        train_dataset = News(root='./data',
                             noise_type=args.noise_type,
                             noise_rate=args.noise_rate,
                             outlier_rate=args.outlier_rate,
                             imbalance_coefficient=args.imbalance_coefficient,
                             alpha=args.alpha,
                             topk_id_classes=args.num_classes
                             )
    elif args.dataset in ['Zircon3', 'Apatite12', 'Garnet7', 'Garnet5', 'Basalt6']:
        config = {
            'Zircon3': (3, 16),
            'Apatite12': (12, 9),
            'Basalt6': (6, 19),
            'Garnet7': (7, 8),
            'Garnet5': (5, 8),
        }
        num_classes, input_features = config[args.dataset]
        args.num_classes = args.output = num_classes
        args.input_features = input_features
        args.input_channel = 1  # 1 / 3
        args.hidden = args.hidden_dim[-1]
        args.network = 'MLPs'
        train_dataset = GeoDataset(root='./data',
                                   dataset_name=args.dataset,
                                   alpha=args.alpha,
                                   data_mode='raw',
                                   )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True,
                                               )
    args.train_dataset = train_dataset
    args.train_loader = train_loader
    args.train_noisy_labels = noisy_labels = train_dataset.train_noisy_labels.copy()
    args.train_clean_labels = clean_labels = train_dataset.train_labels.copy()

    return train_dataset, train_loader, noisy_labels, clean_labels
