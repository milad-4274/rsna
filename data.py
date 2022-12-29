from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os

def create_dataset(path,size_transforms, augment):
    if augment:
        final_transforms = transforms.Compose([
            size_transforms,
            transforms.AutoAugmnet()
        ])
    else:
        final_transforms = size_transforms
    train_datset = ImageFolder(path, transform= final_transforms, target_transform=None)


def create_dataloader(dataset, batch_size):
    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=os.cpu_count()
    )
    return data_loader

def dicom_collate_fn(batch):
    """
       batch: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, labels, lengths = zip(*batch)
    print(labels)
    print(lengths)
    return None
