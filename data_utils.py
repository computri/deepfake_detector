import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

import os
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
import re


def extract_object_class(filename):
    match = re.search(r"\((\d+)\)", filename)
    if match:
        return int(match.group(1))
    else:
        return 0  # fallback or special value if no match

def get_datamodule(cfg, image_size):
    if cfg.dataset_name == "cifake":
        return CIFAKEDataModule(
            data_dir=cfg.root_dir, 
            batch_size=cfg.batch_size, 
            num_workers=cfg.num_workers, 
            image_size=image_size
        )
    elif cfg.dataset_name == "coco":
        return COCODataModule(
            data_dir=cfg.root_dir, 
            batch_size=cfg.batch_size, 
            num_workers=cfg.num_workers, 
            image_size=image_size
        )
    elif cfg.dataset_name == "generated":
        return GeneratedDataModule(
            data_dir=cfg.root_dir, 
            batch_size=cfg.batch_size, 
            num_workers=cfg.num_workers, 
            image_size=image_size
        )
    elif cfg.dataset_name == "mixed":
        return MixedDataModule(
            data_dir=cfg.root_dir, 
            batch_size=cfg.batch_size, 
            num_workers=cfg.num_workers, 
            image_size=image_size
        )


class GeneratedDataset(VisionDataset):
    def __init__(self, root="./data", train=True, transform=None, target_transform=None):
        root = os.path.join(root, "generated_images")
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.data_dir = root
        
        self.dataset = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                path = os.path.join(self.data_dir, fname)
                label = 1.0  # Assuming all images in generated dataset are fake
                samples.append((path, label))
        return samples

    def __getitem__(self, index):
        path, target = self.dataset[index]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, float(target)
    
    def __len__(self):
        return len(self.dataset)
    

class GeneratedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64, num_workers=4, image_size=32):
        super().__init__()
        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.247, 0.243, 0.261)
            )
        ])

    def setup(self, stage=None):
        train_full = GeneratedDataset(self.data_dir, transform=self.transform)

        num_train = len(train_full)

        num_val = int(0.1 * num_train)

        num_train -= num_val

        self.train_dataset, self.val_set = random_split(train_full, [num_train, num_val])
        self.test_dataset = self.val_set

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
    


class CIFAKEDataset(VisionDataset):
    def __init__(self, root="./data", split='train', transform=None, target_transform=None):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        root = os.path.join(root, "cifake")
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.data_dir = os.path.join(root, split)
        self.classes = ['REAL', 'FAKE']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.dataset = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith(".jpg") or fname.endswith(".png"):
                    path = os.path.join(class_dir, fname)
                    label = self.class_to_idx[class_name]
                    samples.append((path, label))
        return samples

    def __getitem__(self, index):
        path, target = self.dataset[index]
        filename = os.path.basename(path)
        object_class = extract_object_class(filename)

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, float(target), object_class
    
    def __len__(self):
        return len(self.dataset)
    

class CIFAKEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64, num_workers=4, image_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.247, 0.243, 0.261)
            )
        ])

    def setup(self, stage=None):
        train_full = CIFAKEDataset(self.data_dir, split='train', transform=self.transform)

        num_train = len(train_full)

        num_val = int(0.1 * num_train)

        num_train -= num_val

        self.train_dataset, self.val_set = random_split(train_full, [num_train, num_val])
        self.test_dataset = CIFAKEDataset(self.data_dir, split='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
    


class CocoReal(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        transform,
        train=True,
        target_transform=None,
        transforms=None,
    ):  
        root = os.path.join(root, "coco")
        super().__init__(os.path.join(root, f"{'train' if train else 'val'}2017"), transforms, transform, target_transform)
        from pycocotools.coco import COCO

        
        ann_file = os.path.join(root, "annotations", f"instances_{'train' if train else 'val'}2017.json")

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        obj_class = self._load_target(id)

        # if self.transforms is not None:
        #     image, obj_class = self.transforms(image, obj_class)

        if self.transform is not None:
            image = self.transform(image)
        return image, 0.0 # all images are real, so we return 0.0 as the label

    def __len__(self):
        return len(self.ids)




class COCODataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="./data",
        batch_size=64,
        num_workers=4,
        image_size=224,
    ):
        super().__init__()
        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup(self, stage=None):


        full_train_dataset = CocoReal(root=self.data_dir, train=True, transform=self.transform)
        self.val_size = int(len(full_train_dataset) * 0.1)
        self.train_size = len(full_train_dataset) - self.val_size

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [self.train_size, self.val_size]
        )

        self.test_dataset = CocoReal(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
    

def get_dataset(dataset_name, root="./data", train=True, transform=None, target_transform=None):
    if dataset_name == "cifake":
        return CIFAKEDataset(root=root, split='train' if train else 'test', transform=transform, target_transform=target_transform)
    elif dataset_name == "coco":
        return CocoReal(root=root, train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == "generated":
        return GeneratedDataset(root=root, train=train, transform=transform, target_transform=target_transform)
    elif dataset_name == "mixed":
        return MixedDataset(root=root, train=train, transform=transform, target_transform=target_transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

class MixedDataset(VisionDataset):
    def __init__(self, root="./data", real_dataset="coco", fake_dataset="generated", train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.data_dir = root

        self.real_dataset = get_dataset(real_dataset, root=root, train=train, transform=self.transform)
        self.fake_dataset = get_dataset(fake_dataset, root=root, train=train, transform=self.transform)

        self.total_size = len(self.real_dataset) + len(self.fake_dataset)


    def __getitem__(self, index):

        if index < len(self.real_dataset):
            # Get real image
            image, target = self.real_dataset[index]
        else:
            # Get fake image
            image, target = self.fake_dataset[index - len(self.real_dataset)]

        return image, target
    
    def __len__(self):
        return self.total_size
    

class MixedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="./data",
        batch_size=64,
        num_workers=4,
        image_size=224,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup(self, stage=None):

        full_train_dataset = MixedDataset(root=self.data_dir, train=True, transform=self.transform)
        self.val_size = int(len(full_train_dataset) * 0.1)
        self.train_size = len(full_train_dataset) - self.val_size

        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [self.train_size, self.val_size]
        )

        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)