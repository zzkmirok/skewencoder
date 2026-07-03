import numpy as np
import torch
import pytest

from skewencoder.io import create_dataset_from_descriptors
from mlcolvar.data import DictDataset, DictModule


class TestCreateDatasetFromDescriptors:

    def test_basic_shape(self):
        descriptors = np.random.randn(50, 16)
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            descriptors, verbose=False
        )
        assert AE_dataset["data"].shape == (50, 16)

    def test_skew_defaults_to_same_data(self):
        descriptors = np.random.randn(50, 16)
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            descriptors, verbose=False
        )
        assert torch.equal(skew_dataset["data"], AE_dataset["data"])

    def test_separate_skew_descriptors(self):
        all_descriptors = np.random.randn(50, 16)
        latest_descriptors = all_descriptors[-10:]
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            all_descriptors, skew_descriptors=latest_descriptors, verbose=False
        )
        assert AE_dataset["data"].shape == (50, 16)
        assert skew_dataset["data"].shape == (10, 16)
        assert torch.allclose(
            skew_dataset["data"], torch.Tensor(latest_descriptors)
        )

    def test_auto_feature_names(self):
        descriptors = np.random.randn(10, 4)
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            descriptors, verbose=False
        )
        assert list(AE_dataset.feature_names) == ["desc_0", "desc_1", "desc_2", "desc_3"]
        assert list(skew_dataset.feature_names) == ["desc_0", "desc_1", "desc_2", "desc_3"]

    def test_custom_feature_names(self):
        descriptors = np.random.randn(10, 3)
        names = ["feat_a", "feat_b", "feat_c"]
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            descriptors, feature_names=names, verbose=False
        )
        assert list(AE_dataset.feature_names) == names

    def test_returns_dict_dataset(self):
        descriptors = np.random.randn(20, 8)
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            descriptors, verbose=False
        )
        assert isinstance(AE_dataset, DictDataset)
        assert isinstance(skew_dataset, DictDataset)
        assert "data" in AE_dataset.keys
        assert "data" in skew_dataset.keys

    def test_returns_dict_module(self):
        descriptors = np.random.randn(20, 8)
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            descriptors, verbose=False
        )
        assert isinstance(datamodule, DictModule)

    def test_datamodule_iterable(self):
        descriptors = np.random.randn(30, 5)
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            descriptors, batch_size=10, verbose=False
        )
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        assert "dataset0" in batch
        assert "data" in batch["dataset0"]

    def test_batch_size_scaled_by_multiple(self):
        all_descriptors = np.random.randn(60, 5)
        latest_descriptors = all_descriptors[-20:]
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            all_descriptors,
            skew_descriptors=latest_descriptors,
            multiple=3,
            batch_size=5,
            verbose=False,
        )
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        assert batch["dataset0"]["data"].shape[0] == 15
        assert batch["dataset1"]["data"].shape[0] == 5

    def test_rejects_1d_input(self):
        descriptors = np.random.randn(50)
        with pytest.raises(ValueError, match="must be a 2D array"):
            create_dataset_from_descriptors(descriptors, verbose=False)

    def test_rejects_3d_input(self):
        descriptors = np.random.randn(10, 5, 3)
        with pytest.raises(ValueError, match="must be a 2D array"):
            create_dataset_from_descriptors(descriptors, verbose=False)

    def test_rejects_1d_skew_input(self):
        descriptors = np.random.randn(50, 16)
        skew_descriptors = np.random.randn(10)
        with pytest.raises(ValueError, match="skew_descriptors must be a 2D array"):
            create_dataset_from_descriptors(
                descriptors, skew_descriptors=skew_descriptors, verbose=False
            )

    def test_rejects_feature_mismatch(self):
        descriptors = np.random.randn(50, 16)
        skew_descriptors = np.random.randn(10, 8)
        with pytest.raises(ValueError, match="same number of features"):
            create_dataset_from_descriptors(
                descriptors, skew_descriptors=skew_descriptors, verbose=False
            )

    def test_tensor_dtype_float(self):
        descriptors = np.random.randn(10, 4).astype(np.float64)
        AE_dataset, _, _ = create_dataset_from_descriptors(descriptors, verbose=False)
        assert AE_dataset["data"].dtype == torch.float32

    def test_single_sample(self):
        descriptors = np.random.randn(1, 8)
        AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
            descriptors, verbose=False
        )
        assert AE_dataset["data"].shape == (1, 8)
