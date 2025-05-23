# ultimate_morph_generator/data_management/dataset_utils.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision import datasets, transforms  # type: ignore
from PIL import Image  # type: ignore
from typing import List, Tuple, Optional, Callable

from ..config import get_config, DataManagementConfig, DGOOracleConfig
from ..utilities.image_utils import standardize_image, pil_to_cv, cv_to_pil, preprocess_image_for_dgo
from ..utilities.type_definitions import CvImage, Label, ImagePath
from ..utilities.logging_config import setup_logging

logger = setup_logging()


class InitialSamplesDataset(Dataset):
    """
    A PyTorch Dataset for loading initial character samples from a directory.
    """

    def __init__(self, image_paths: List[ImagePath], labels: List[Label],
                 target_size: Tuple[int, int], grayscale: bool,
                 transform: Optional[Callable] = None, use_pil_preprocessing: bool = True):
        """
        Args:
            image_paths (List[ImagePath]): List of paths to images.
            labels (List[Label]): List of corresponding labels.
            target_size (Tuple[int, int]): Desired (H, W) for the images.
            grayscale (bool): Whether to convert images to grayscale.
            transform (Optional[Callable]): PyTorch transforms to apply after loading and basic processing.
            use_pil_preprocessing (bool): If True, uses PIL for loading and initial transforms (resize, grayscale).
                                         If False, uses OpenCV for loading and standardization.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size
        self.grayscale = grayscale
        self.transform = transform
        self.use_pil_preprocessing = use_pil_preprocessing

        if len(image_paths) != len(labels):
            raise ValueError("Number of image paths must match number of labels.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Label]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            if self.use_pil_preprocessing:
                # PIL based loading and preprocessing (similar to preprocess_image_for_dgo's PIL part)
                pil_img = Image.open(img_path)

                # Basic PIL transformations to match DGO input expectations somewhat
                pil_transform_list = []
                if self.grayscale:
                    pil_transform_list.append(transforms.Grayscale(num_output_channels=1))
                pil_transform_list.append(
                    transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BICUBIC))

                # Apply main PyTorch transform (which should include ToTensor and Normalize)
                # If self.transform is None, we need at least ToTensor
                if self.transform:
                    # Apply basic PIL transforms first
                    intermediate_pil_img = transforms.Compose(pil_transform_list)(
                        pil_img) if pil_transform_list else pil_img
                    tensor_img = self.transform(intermediate_pil_img)
                else:
                    # Default minimal transform if none provided (ToTensor)
                    pil_transform_list.append(transforms.ToTensor())
                    tensor_img = transforms.Compose(pil_transform_list)(pil_img)

            else:  # OpenCV based loading
                # Load with OpenCV, standardize (resize, grayscale, optionally invert)
                cv_img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load as is first
                if cv_img_raw is None:
                    raise IOError(f"Could not read image: {img_path}")

                # Use standardize_image for consistent processing
                cv_img_std = standardize_image(
                    cv_img_raw,
                    target_size=self.target_size,
                    grayscale=self.grayscale,
                    invert_if_dark_on_light=True  # Common for character data
                )  # Returns uint8 numpy array

                # Convert to PIL to apply PyTorch transforms (if any)
                pil_img_from_cv = cv_to_pil(cv_img_std)

                if self.transform:
                    tensor_img = self.transform(pil_img_from_cv)
                else:  # Default ToTensor if no transform specified
                    tensor_img = transforms.ToTensor()(pil_img_from_cv)

        except Exception as e:
            logger.error(f"Error loading or processing image {img_path}: {e}")
            # Return a dummy tensor and label to prevent Dataloader crash
            # Or, re-raise and handle in dataloader's collate_fn or sampler
            dummy_channels = 1 if self.grayscale else 3
            tensor_img = torch.zeros((dummy_channels, self.target_size[0], self.target_size[1]))
            # label remains, or use a special error label if your training loop handles it

        return tensor_img, label


def load_initial_char_samples(char_string: str,
                              target_label: int,
                              max_samples: Optional[int] = None) -> Tuple[List[ImagePath], List[Label]]:
    """
    Loads initial samples for a specific character from the configured path.
    Returns a list of image paths and a list of corresponding labels.
    """
    cfg = get_config()
    data_cfg = cfg.data_management

    char_path_str = data_cfg.initial_samples_path_template.format(char_string=char_string)

    image_paths: List[ImagePath] = []
    labels: List[Label] = []

    if not os.path.exists(char_path_str) or not os.listdir(char_path_str):
        logger.warning(f"Initial sample path for char '{char_string}' ({char_path_str}) is empty or does not exist.")
        logger.info("Attempting to download/generate a few MNIST samples as fallback if target_label is a digit.")

        if 0 <= target_label <= 9:
            try:
                mnist_data_path = './temp_mnist_data'
                # Ensure directory for MNIST data exists
                os.makedirs(mnist_data_path, exist_ok=True)
                # Ensure target directory for initial samples exists
                os.makedirs(char_path_str, exist_ok=True)

                mnist_dataset = datasets.MNIST(mnist_data_path, train=True, download=True)
                count = 0
                for i in range(len(mnist_dataset)):
                    img, label_val = mnist_dataset[i]  # img is PIL Image
                    if label_val == target_label:
                        # Save PIL image to the initial_samples_path
                        # Convert to CvImage then standardize (size, grayscale etc.) before saving
                        # This ensures even downloaded samples are consistent.
                        cv_img_from_pil = pil_to_cv(img)
                        standardized_for_save = standardize_image(
                            cv_img_from_pil,
                            target_size=data_cfg.target_image_size,
                            grayscale=data_cfg.grayscale_input,
                            invert_if_dark_on_light=True  # MNIST is black on white, usually we want white on black
                        )

                        filename = f"mnist_fallback_{char_string}_{i}{data_cfg.image_file_format}"
                        save_path = os.path.join(char_path_str, filename)
                        cv2.imwrite(save_path, standardized_for_save)

                        image_paths.append(ImagePath(save_path))
                        labels.append(target_label)
                        count += 1
                        if max_samples is not None and count >= max_samples:
                            break
                        if count >= 10:  # Get up to 10 fallback samples
                            break
                if count > 0:
                    logger.info(
                        f"Downloaded and saved {count} MNIST samples for char '{char_string}' to {char_path_str}")
                else:
                    logger.warning(f"Could not find/download MNIST samples for label {target_label}.")

                # Clean up temp MNIST download dir
                # import shutil
                # shutil.rmtree(mnist_data_path, ignore_errors=True)

            except Exception as e:
                logger.error(f"Error during MNIST fallback sample generation: {e}")
        else:
            logger.error(f"Cannot use MNIST fallback for non-digit target_label: {target_label}")

        if not image_paths:  # If still no paths after fallback
            logger.error("No initial samples found or generated. Critical for DGO training.")
            # Consider raising an exception if no initial samples are absolutely required
            # raise FileNotFoundError(f"No initial samples for char '{char_string}' and fallback failed.")
            return [], []  # Return empty lists

    else:  # Path exists and is not empty
        for fname in sorted(os.listdir(char_path_str)):  # Sort for consistency
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                full_path = os.path.join(char_path_str, fname)
                image_paths.append(ImagePath(full_path))
                labels.append(target_label)  # All samples in this folder belong to target_label
                if max_samples is not None and len(image_paths) >= max_samples:
                    break
        logger.info(f"Loaded {len(image_paths)} initial samples for char '{char_string}' from {char_path_str}")

    return image_paths, labels


def get_dgo_pytorch_transform(dgo_cfg: DGOOracleConfig, data_cfg: DataManagementConfig,
                              augment: bool = False) -> Callable:
    """
    Creates the PyTorch transform pipeline for DGO training/evaluation.
    """
    transform_list = []

    # Augmentations (only if augment=True, typically for training)
    if augment:
        # Keep augmentations mild not to deviate too much from learned DGO features
        transform_list.append(transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3))
        transform_list.append(transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            # transforms.ElasticTransform(alpha=10.0, sigma=2.0) # Might be too strong/slow
        ]))
        transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))  # If not grayscale

    # Basic transforms (applied always, after augmentations if any)
    # Note: InitialSamplesDataset already handles resize and grayscale if use_pil_preprocessing=True.
    # If use_pil_preprocessing=False in dataset, then these need to be here or ensure consistency.
    # For DGO, we generally assume input is already sized and grayscaled as per config.
    # ToTensor() is crucial.
    transform_list.append(
        transforms.ToTensor())  # Converts PIL [0,255] (or CV [0,1] if from standardized) to Tensor [0,1]

    # Normalization (should match DGO's training)
    # Example:
    if data_cfg.grayscale_input:
        # transform_list.append(transforms.Normalize((0.1307,), (0.3081,))) # MNIST-like norm
        # Or, if DGO is trained on [0,1] without specific norm:
        pass
    else:  # Color
        # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        pass

    return transforms.Compose(transform_list)


def create_dgo_dataloader(image_label_pairs: List[Tuple[Union[ImagePath, CvImage], Label]],
                          dgo_cfg: DGOOracleConfig,
                          data_cfg: DataManagementConfig,
                          batch_size: int,
                          shuffle: bool = True,
                          augment: bool = False,
                          num_workers: int = 0,
                          pin_memory: bool = True,
                          use_weighted_sampler: bool = False) -> DataLoader:
    """
    Creates a PyTorch DataLoader for DGO training/finetuning from a list of (image or path, label) pairs.
    """
    # Separate paths and actual image data if mixed
    img_paths_or_data = [pair[0] for pair in image_label_pairs]
    labels = [pair[1] for pair in image_label_pairs]

    # Determine if input is paths or direct CvImage data
    # This is a bit tricky. For simplicity, assume if first item is string, all are paths.
    # A more robust way would be to check each item type.
    # Here, InitialSamplesDataset expects paths, so we need to adapt if direct CvImage data is passed.

    # For now, assume `InitialSamplesDataset` is flexible or we adapt the input.
    # If `image_label_pairs` contains CvImage, `InitialSamplesDataset` needs modification
    # or we use a different dataset class like `TensorDataset` after preprocessing.

    # Let's assume image_label_pairs contains ImagePath for InitialSamplesDataset.
    # If they are CvImage, they'd need to be saved to temp files or use a different Dataset.

    # This function is generic; the dataset used should handle the input type.
    # We'll use InitialSamplesDataset assuming paths are provided.
    # If direct CvImages are provided for finetuning, they should be preprocessed into Tensors first.

    pytorch_transform = get_dgo_pytorch_transform(dgo_cfg, data_cfg, augment=augment)

    # We use use_pil_preprocessing=True in dataset so it handles initial resize/grayscale
    # The pytorch_transform then does ToTensor and Normalize (+ augmentations)
    dataset = InitialSamplesDataset(
        image_paths=img_paths_or_data,  # Requires this to be List[ImagePath]
        labels=labels,
        target_size=data_cfg.target_image_size,
        grayscale=data_cfg.grayscale_input,
        transform=pytorch_transform,
        use_pil_preprocessing=True  # Consistent with how DGO expects input
    )

    sampler = None
    if use_weighted_sampler and len(dataset) > 0:
        # Calculate weights for each sample for balancing (if needed)
        class_counts = np.bincount(labels, minlength=dgo_cfg.num_classes)
        # Avoid division by zero if a class has no samples
        # Weights are 1 / count_of_class_for_this_sample
        sample_weights = [1.0 / class_counts[label] if class_counts[label] > 0 else 0 for label in labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False  # Sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle or sampler is not None)  # Drop last if shuffling to avoid smaller batch
    )
    return dataloader


if __name__ == "__main__":
    # --- Test dataset_utils ---
    from ..config import SystemConfig

    temp_cfg_data = {
        "project_name": "DatasetUtilsTest",
        "target_character_string": "test_char_digit",  # For MNIST fallback
        "target_character_index": 7,  # Corresponds to '7' for MNIST fallback
        "data_management": {
            "initial_samples_path_template": "./temp_initial_samples/char_{char_string}/",
            "target_image_size": (28, 28),
            "grayscale_input": True,
            "image_file_format": ".png"
        },
        "dgo_oracle": {"num_classes": 10},  # For weighted sampler
        "logging": {"level": "DEBUG"}
    }
    from ..config import _config_instance

    _config_instance = SystemConfig.model_validate(temp_cfg_data)

    logger = setup_logging()

    # Test load_initial_char_samples (will try MNIST fallback)
    test_char_str = get_config().target_character_string
    test_char_idx = get_config().target_character_index

    # Ensure temp initial samples dir exists for the test
    os.makedirs(get_config().data_management.initial_samples_path_template.format(char_string=test_char_str),
                exist_ok=True)

    img_paths, lbls = load_initial_char_samples(test_char_str, test_char_idx, max_samples=5)
    print(
        f"\nLoaded {len(img_paths)} initial samples for '{test_char_str}'. First path: {img_paths[0] if img_paths else 'None'}")
    assert len(img_paths) > 0, "MNIST fallback should have provided samples"
    assert all(label == test_char_idx for label in lbls)

    # Test create_dgo_dataloader
    if img_paths:
        dgo_loader = create_dgo_dataloader(
            image_label_pairs=list(zip(img_paths, lbls)),
            dgo_cfg=get_config().dgo_oracle,
            data_cfg=get_config().data_management,
            batch_size=2,
            shuffle=True,
            augment=True,  # Test with augmentations
            use_weighted_sampler=True  # Test with sampler
        )
        print(f"Created DataLoader with {len(dgo_loader.dataset)} samples.")

        # Iterate over a few batches
        for i, (batch_images, batch_labels) in enumerate(dgo_loader):
            print(f"Batch {i + 1}: Images shape {batch_images.shape}, Labels {batch_labels}")
            assert batch_images.ndim == 4
            # channels = 1 if get_config().data_management.grayscale_input else 3
            # H, W = get_config().data_management.target_image_size
            # assert batch_images.shape[1:] == (channels, H, W)
            if i >= 1:  # Check a couple of batches
                break
        assert i >= 0, "Dataloader did not produce batches"

    print("\nDataset_utils tests completed.")
    # Clean up temp dirs
    # import shutil
    # shutil.rmtree("./temp_initial_samples/", ignore_errors=True)
    # shutil.rmtree("./temp_mnist_data/", ignore_errors=True)