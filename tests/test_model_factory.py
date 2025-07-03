import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.model_factory import get_model, SUPPORTED_MODELS


class TestModelFactory:
    """Test suite for model factory functionality."""

    MODEL_STARTPOINTS = {
        "regnet_y_16gf": "trunk_output.block3",
        "resnet50": "layer4",
        "resnext50_32x4d": "layer3",
        "resnext101_32x8d": "layer3",
        "resnext101_64x4d": "layer3",
        "convnext_base": "features.2",
        "convnext_large": "features.2",
        "convnext_small": "features.6",
        "efficientnet_v2_l": "features.7.6",
        "efficientnet_v2_m": "features.7.2",
        "efficientnet_v2_s": "features.6.14",
        "cnn_avg": None,
        "cnn_fc": None,
    }

    @pytest.fixture
    def device(self):
        """Fixture for device selection."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def n_classes(self):
        """Fixture for number of classes."""
        return 10

    @pytest.fixture
    def dummy_data(self):
        """Fixture for dummy normalization statistics."""
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    @pytest.fixture
    def dummy_dataloader(self, device):
        """Fixture for creating a dummy dataloader for training tests."""
        # Create dummy data: batch_size=4, channels=3, height=224, width=224
        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_labels = torch.randint(0, 10, (batch_size,))

        dataset = TensorDataset(dummy_images, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def create_base_config(self, architecture):
        """Create a basic configuration for testing."""
        startpoint = self.MODEL_STARTPOINTS.get(architecture)
        config = {
            "architecture": architecture,
            "model": {
                "startpoint": startpoint,
                "dropout": 0.2,
                "last_filter_size": 256,
                "alpha": 0.2,
                "input_size": 224,
                "hidden_size": 512,
            },
        }
        return DictConfig(config)

    @pytest.mark.parametrize("architecture", SUPPORTED_MODELS)
    def test_model_instantiation(self, architecture, device, n_classes, dummy_data):
        """Test that all supported models can be instantiated successfully."""
        mean, std = dummy_data
        cfg = self.create_base_config(architecture)

        try:
            model, transforms, returned_mean, returned_std = get_model(
                cfg, device, mean, std, n_classes
            )

            # Basic assertions
            assert model is not None, f"Model {architecture} should not be None"
            assert isinstance(model, nn.Module), (
                f"Model {architecture} should be a nn.Module"
            )
            assert transforms is not None, (
                f"Transforms for {architecture} should not be None"
            )
            assert returned_mean is not None, (
                f"Mean for {architecture} should not be None"
            )
            assert returned_std is not None, (
                f"Std for {architecture} should not be None"
            )

            # Check model is on correct device
            assert next(model.parameters()).device.type == device.split(":")[0], (
                f"Model {architecture} should be on {device}"
            )

            print(f"✓ Successfully instantiated {architecture}")

        except Exception as e:
            pytest.fail(f"Failed to instantiate model {architecture}: {str(e)}")

    @pytest.mark.parametrize("architecture", SUPPORTED_MODELS)
    def test_model_forward_pass(self, architecture, device, n_classes, dummy_data):
        """Test that all models can perform a forward pass."""
        mean, std = dummy_data
        cfg = self.create_base_config(architecture)

        model, transforms, _, _ = get_model(cfg, device, mean, std, n_classes)

        # Create dummy input
        batch_size = 2
        # All models expect 4D input (batch, channels, height, width)
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

        try:
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)

            # Check output shape
            assert output.shape[0] == batch_size, (
                f"Batch dimension mismatch for {architecture}"
            )
            assert output.shape[1] == n_classes, (
                f"Output classes mismatch for {architecture}"
            )

            print(f"✓ Forward pass successful for {architecture}")

        except Exception as e:
            pytest.fail(f"Forward pass failed for {architecture}: {str(e)}")

    @pytest.mark.parametrize("architecture", SUPPORTED_MODELS)
    def test_single_training_step(
        self, architecture, device, n_classes, dummy_data, dummy_dataloader
    ):
        """Test a single training step for each model."""
        mean, std = dummy_data
        cfg = self.create_base_config(architecture)

        model, transforms, _, _ = get_model(cfg, device, mean, std, n_classes)

        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        try:
            model.train()

            # Get a batch of data
            for batch_images, batch_labels in dummy_dataloader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                # All models expect 4D input, no special handling needed

                # Forward pass
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Verify loss is a scalar and finite
                assert loss.item() > 0, f"Loss should be positive for {architecture}"
                assert torch.isfinite(loss), f"Loss should be finite for {architecture}"

                print(
                    f"✓ Training step successful for {architecture}, loss: {loss.item():.4f}"
                )
                break  # Only test one batch

        except Exception as e:
            pytest.fail(f"Training step failed for {architecture}: {str(e)}")

    def test_unsupported_model_raises_error(self, device, n_classes, dummy_data):
        """Test that unsupported model architecture raises ValueError."""
        mean, std = dummy_data
        cfg = self.create_base_config("unsupported_model")

        with pytest.raises(ValueError, match="Unkown model"):
            get_model(cfg, device, mean, std, n_classes)

    @pytest.mark.parametrize(
        "architecture", ["resnet50", "efficientnet_v2_m", "convnext_small"]
    )
    def test_pretrained_models_frozen_correctly(
        self, architecture, device, n_classes, dummy_data
    ):
        """Test that pretrained models have parameters frozen correctly."""
        mean, std = dummy_data
        cfg = self.create_base_config(architecture)
        cfg.model.startpoint = "features.0"  # Early layer to test freezing

        model, _, _, _ = get_model(cfg, device, mean, std, n_classes)

        # Check that some parameters are frozen and some are not
        frozen_params = []
        trainable_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)

        # Should have both frozen and trainable parameters
        assert len(frozen_params) > 0, (
            f"Model {architecture} should have some frozen parameters"
        )
        assert len(trainable_params) > 0, (
            f"Model {architecture} should have some trainable parameters"
        )

        print(
            f"✓ {architecture} has {len(frozen_params)} frozen and {len(trainable_params)} trainable parameters"
        )

    @pytest.mark.parametrize("architecture", ["cnn_avg", "cnn_fc"])
    def test_custom_models_all_trainable(
        self, architecture, device, n_classes, dummy_data
    ):
        """Test that custom models have all parameters trainable."""
        mean, std = dummy_data
        cfg = self.create_base_config(architecture)

        model, _, _, _ = get_model(cfg, device, mean, std, n_classes)

        # All parameters should be trainable for custom models
        for name, param in model.named_parameters():
            assert param.requires_grad, (
                f"Parameter {name} in {architecture} should be trainable"
            )

        print(f"✓ All parameters in {architecture} are trainable")

    def test_transforms_returned_correctly(self, device, n_classes, dummy_data):
        """Test that transforms are returned in the correct format."""
        mean, std = dummy_data
        cfg = self.create_base_config("resnet50")

        _, transforms, _, _ = get_model(cfg, device, mean, std, n_classes)

        # Should return a dictionary with transforms
        assert isinstance(transforms, dict), "Transforms should be a dictionary"
        assert "test" in transforms, "Transforms should contain 'test' key"
        assert "train_hard" in transforms, "Transforms should contain 'train_hard' key"

        print("✓ Transforms returned in correct format")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
