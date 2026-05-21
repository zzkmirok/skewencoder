import numpy as np
import torch
import pytest

from skewencoder.io import create_dataset_from_descriptors
from mlcolvar.data import DictDataset, DictModule


class TestCreateDatasetFromDescriptors:

    def test_basic_shape(self):
        descriptors = np.random.randn(50, 16)
        dataset, datamodule = create_dataset_from_descriptors(descriptors, verbose=False)
        assert dataset["data"].shape == (50, 16)

    def test_auto_feature_names(self):
        descriptors = np.random.randn(10, 4)
        dataset, datamodule = create_dataset_from_descriptors(descriptors, verbose=False)
        assert list(dataset.feature_names) == ["desc_0", "desc_1", "desc_2", "desc_3"]

    def test_custom_feature_names(self):
        descriptors = np.random.randn(10, 3)
        names = ["feat_a", "feat_b", "feat_c"]
        dataset, datamodule = create_dataset_from_descriptors(
            descriptors, feature_names=names, verbose=False
        )
        assert list(dataset.feature_names) == names

    def test_returns_dict_dataset(self):
        descriptors = np.random.randn(20, 8)
        dataset, datamodule = create_dataset_from_descriptors(descriptors, verbose=False)
        assert isinstance(dataset, DictDataset)
        assert "data" in dataset.keys

    def test_returns_dict_module(self):
        descriptors = np.random.randn(20, 8)
        dataset, datamodule = create_dataset_from_descriptors(descriptors, verbose=False)
        assert isinstance(datamodule, DictModule)

    def test_datamodule_iterable(self):
        descriptors = np.random.randn(30, 5)
        dataset, datamodule = create_dataset_from_descriptors(
            descriptors, batch_size=10, verbose=False
        )
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        assert "dataset0" in batch
        assert "data" in batch["dataset0"]

    def test_rejects_1d_input(self):
        descriptors = np.random.randn(50)
        with pytest.raises(ValueError, match="must be a 2D array"):
            create_dataset_from_descriptors(descriptors, verbose=False)

    def test_rejects_3d_input(self):
        descriptors = np.random.randn(10, 5, 3)
        with pytest.raises(ValueError, match="must be a 2D array"):
            create_dataset_from_descriptors(descriptors, verbose=False)

    def test_tensor_dtype_float(self):
        descriptors = np.random.randn(10, 4).astype(np.float64)
        dataset, _ = create_dataset_from_descriptors(descriptors, verbose=False)
        assert dataset["data"].dtype == torch.float32

    def test_single_sample(self):
        descriptors = np.random.randn(1, 8)
        dataset, datamodule = create_dataset_from_descriptors(descriptors, verbose=False)
        assert dataset["data"].shape == (1, 8)
