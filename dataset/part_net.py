import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import print_log


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    """
    Dataset class for ShapeNet data. Inherits from PyTorch's Dataset class.

    :param config: Configuration object with dataset parameters.
    """
    def __init__(self, config) -> None:
        self.data_root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset  # train, test, val

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        self.test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = config.npoint
        self.whole = config.get('whole', False)

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='PartNet')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='PartNet')
        self._load_file_list()

        self.permutation = np.arange(self.npoints)

    def _load_file_list(self) -> None:
        """
        Load the file list from data files.
        """
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(self.test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {self.test_data_list_file}', logger='ShapeNet-55')
            lines.extend(test_lines)
        self.file_list = [
            {
                'taxonomy_id': line.split('-')[0],
                'model_id': line.split('-')[1].split('.')[0],
                'file_path': line.strip()
            } for line in lines
        ]
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

    def random_sample(self, pc: np.ndarray, num: int) -> np.ndarray:
        """
        Randomly sample points from a point cloud.

        :param pc: The input point cloud.
        :param num: The number of points to sample.
        :return: The sampled points.
        """
        np.random.shuffle(self.permutation)
        return pc[self.permutation[:num]]

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a data sample given its index.

        :param idx: The index of the data sample.
        :return: A tuple containing the taxonomy ID, model ID, and the data.
        """
        sample = self.file_list[idx]
        data = IO.get(os.path.join(self.data_root, sample['file_path'])).astype(np.float32)
        data = self.random_sample(data, self.sample_points_num)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self) -> int:
        """
        Get the total number of data samples in the dataset.

        :return: The total number of data samples.
        """
        return len(self.file_list)
