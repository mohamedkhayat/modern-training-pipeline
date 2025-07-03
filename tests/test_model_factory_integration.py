import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import os
import shutil
from PIL import Image
import numpy as np
from omegaconf import DictConfig
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.model_factory import get_model, SUPPORTED_MODELS
from dataset import DS
from utils.data_utils import get_data_samples


class TestModelFactoryIntegration:
    """Integration tests for model factory with real data flow."""

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

    @pytest.fixture(scope="class")
    def temp_dataset(self):
        """Create a temporary dataset for testing."""
        temp_dir = tempfile.mkdtemp()

        # Create class directories
        classes = ["class_0", "class_1", "class_2"]
        samples_per_class = 5

        for class_name in classes:
            class_dir = os.path.join(temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Create dummy images
            for i in range(samples_per_class):
                # Create a random RGB image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = os.path.join(class_dir, f"image_{i}.jpg")
                img.save(img_path)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def device(self):
        """Fixture for device selection."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def real_dataset(self, temp_dataset):
        """Create a real dataset from temporary images."""
        samples, classes, class_to_idx = get_data_samples(temp_dataset)
        dataset = DS(samples, classes, class_to_idx)
        return dataset, classes, class_to_idx

    def create_config(self, architecture):
        """Create configuration for testing."""
        startpoint = self.MODEL_STARTPOINTS.get(architecture)
        config = {
            "architecture": architecture,
            "batch_size": 2,
            "lr": 0.001,
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

    @pytest.mark.parametrize(
        "architecture", ["resnet50", "efficientnet_v2_s", "convnext_small"]
    )
    def test_end_to_end_training_loop(
        self, architecture, device, real_dataset, temp_dataset
    ):
        """Test complete training loop with real data for key architectures."""
        dataset, classes, class_to_idx = real_dataset
        n_classes = len(classes)
        cfg = self.create_config(architecture)

        # Get model and transforms
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        model, transforms, _, _ = get_model(cfg, device, mean, std, n_classes)

        # Apply transforms to dataset
        dataset.transforms = transforms["test"]  # Use simpler test transforms

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        # Training loop
        model.train()
        total_loss = 0
        num_batches = 0

        try:
            for batch_images, batch_labels in dataloader:
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

                total_loss += loss.item()
                num_batches += 1

                # Test predictions
                _, predicted = torch.max(outputs.data, 1)
                assert predicted.shape == batch_labels.shape, (
                    "Prediction shape mismatch"
                )

                # Only test a few batches to keep tests fast
                if num_batches >= 2:
                    break

            avg_loss = total_loss / num_batches
            assert avg_loss > 0, f"Average loss should be positive for {architecture}"
            assert avg_loss < 100, (
                f"Average loss seems too high for {architecture}: {avg_loss}"
            )

            print(
                f"✓ End-to-end training successful for {architecture}, avg_loss: {avg_loss:.4f}"
            )

        except Exception as e:
            pytest.fail(f"End-to-end training failed for {architecture}: {str(e)}")

    @pytest.mark.parametrize("architecture", SUPPORTED_MODELS)
    def test_model_eval_mode(self, architecture, device, real_dataset):
        """Test that models work correctly in evaluation mode."""
        dataset, classes, class_to_idx = real_dataset
        n_classes = len(classes)
        cfg = self.create_config(architecture)

        # Get model
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        model, transforms, _, _ = get_model(cfg, device, mean, std, n_classes)

        # Apply transforms and create dataloader
        dataset.transforms = transforms["test"]
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        try:
            model.eval()
            with torch.no_grad():
                for batch_images, batch_labels in dataloader:
                    batch_images = batch_images.to(device)

                    # All models expect 4D input, no special handling needed

                    outputs = model(batch_images)

                    # Check output properties
                    assert outputs.shape[0] == 1, "Batch size should be 1"
                    assert outputs.shape[1] == n_classes, (
                        f"Output should have {n_classes} classes"
                    )
                    assert torch.isfinite(outputs).all(), "All outputs should be finite"

                    # Test only one batch
                    break

            print(f"✓ Evaluation mode test passed for {architecture}")

        except Exception as e:
            pytest.fail(f"Evaluation mode test failed for {architecture}: {str(e)}")

    def test_model_parameter_count(self, device, real_dataset):
        """Test that models have reasonable parameter counts."""
        dataset, classes, class_to_idx = real_dataset
        n_classes = len(classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        expected_ranges = {
            "cnn_avg": (1000, 1000000),  # Custom CNN should be smaller
            "cnn_fc": (100000, 10000000),  # FC layers can be large
            "resnet50": (20000000, 30000000),  # ResNet50 is ~25M parameters
            "efficientnet_v2_s": (20000000, 25000000),  # EfficientNet V2 Small
            "efficientnet_v2_m": (50000000, 60000000),  # EfficientNet V2 Medium
        }

        for architecture in ["cnn_avg", "cnn_fc", "resnet50", "efficientnet_v2_s"]:
            cfg = self.create_config(architecture)
            model, _, _, _ = get_model(cfg, device, mean, std, n_classes)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            if architecture in expected_ranges:
                min_params, max_params = expected_ranges[architecture]
                assert min_params <= total_params <= max_params, (
                    f"{architecture} has {total_params} parameters, expected {min_params}-{max_params}"
                )

            assert trainable_params > 0, (
                f"{architecture} should have trainable parameters"
            )

            print(
                f"✓ {architecture}: {total_params:,} total params, {trainable_params:,} trainable"
            )

    def test_gradient_flow(self, device, real_dataset):
        """Test that gradients flow correctly through the models."""
        dataset, classes, class_to_idx = real_dataset
        n_classes = len(classes)

        # Test a few key architectures
        test_architectures = ["resnet50", "efficientnet_v2_s", "cnn_avg"]

        for architecture in test_architectures:
            cfg = self.create_config(architecture)
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            model, transforms, _, _ = get_model(cfg, device, mean, std, n_classes)

            # Create dummy input - all models expect 4D input
            dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)

            dummy_target = torch.randint(0, n_classes, (1,)).to(device)

            # Forward and backward pass
            model.train()
            criterion = nn.CrossEntropyLoss()

            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()

            # Check that gradients exist for trainable parameters
            trainable_params_with_grad = 0
            trainable_params_total = 0

            for param in model.parameters():
                if param.requires_grad:
                    trainable_params_total += 1
                    if param.grad is not None:
                        trainable_params_with_grad += 1

            assert trainable_params_with_grad > 0, (
                f"No gradients computed for {architecture}"
            )
            assert trainable_params_with_grad == trainable_params_total, (
                f"Not all trainable parameters have gradients in {architecture}"
            )

            print(f"✓ Gradient flow verified for {architecture}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
