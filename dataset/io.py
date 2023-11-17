import os
import h5py
import logging
import numpy as np


class IO:
    """
    Serves as a utility for reading data from files with different extensions:
    Supported extensions: .npy, .h5, .txt
    """

    @classmethod
    def get(cls, file_path: str):
        _, file_extension = os.path.splitext(file_path)

        try:
            if file_extension in ['.npy']:
                return cls._read_npy(file_path)
            elif file_extension in ['.h5']:
                return cls._read_h5(file_path)
            else:
                supported_extensions = ['.npy', '.h5']
                raise Exception(f'Unsupported file extension {file_extension}.'
                                f'Supported exstesions: {supported_extensions}')
            
        except Exception as e:
            raise logging.error(f'Error occurred while reading {file_extension} file: {e}')

        
    @classmethod
    def _read_npy(cls, file_path: str) -> np.ndarray:
        """
        Reads a .npy file.

        :param file_path: Path to the .npy file.
        :return: Data as a NumPy array.
        """ 
        try:
            return np.load(file_path)
        except Exception as e:
            raise logging.error(f'Error while reading .npy file {e}')
    
    @classmethod
    def _read_h5(cls, file_path: str, dataset: str = 'data') -> np.ndarray:
        """
        Reads a dataset from an HDF5 file.

        :param file_path: Path to the HDF5 file.
        :param dataset: Name of the dataset to be read from the file. Defaults to 'data'.
        :return: Data from the specified dataset as a NumPy array.
        """        
        try:
            with h5py.File(file_path, 'r') as f:
                if dataset in f:
                    return f[dataset][()]
                else:
                    raise KeyError(f'Dataset {dataset} not in the file')
        except Exception as e:
            raise logging.error(f'Error while reading .h5 file: {e}')
