import torch

class RandomSingleColorReplaceBlack():
    def __init__(self, p=1.0):
        self.probability = p

    def __call__(self, img_tensor):
        if torch.rand(1).item() > self.probability:
            return img_tensor
        random_color = torch.rand(3, device=img_tensor.device)
        black_pixels_mask = torch.all(img_tensor <= 0.01, dim=0)
        for c in range(3):
            img_tensor[c][black_pixels_mask] = random_color[c]
        return img_tensor

class RandomSingleColorReplaceNonBlack():
    """Replaces non-black pixels with a single random color."""
    def __init__(self, p=1.0):
        self.probability = p

    def __call__(self, img_tensor):
        if torch.rand(1).item() > self.probability:
            return img_tensor
        random_color = torch.rand(3, device=img_tensor.device)
        non_black_pixels_mask = torch.any(img_tensor > 0.01, dim=0)
        for c in range(3):
            img_tensor[c][non_black_pixels_mask] = random_color[c]
        return img_tensor


class RandomSingleColorReplaceAll():
    """Replaces all pixels with a single random color, ensuring output is always 3 channels."""
    def __init__(self, p=1.0):
        self.probability = p

    def __call__(self, img_tensor):
        # img_tensor is expected to be a PyTorch tensor with shape [1, 28, 28]
        
        # Convert to 3-channel if it's a single-channel grayscale image
        if img_tensor.size(0) == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)  # Convert to 3 channels [3, 28, 28]
        
        if torch.rand(1).item() <= self.probability:
            # Apply the random color replacement
            black_replacement_color = torch.rand(3, device=img_tensor.device)
            non_black_replacement_color = torch.rand(3, device=img_tensor.device)
            
            # Create masks for black and non-black pixels
            black_pixels_mask = torch.all(img_tensor <= 0.01, dim=0)
            non_black_pixels_mask = ~black_pixels_mask

            # Replace black and non-black pixels with the respective colors
            for c in range(3):
                img_tensor[c][black_pixels_mask] = black_replacement_color[c]
                img_tensor[c][non_black_pixels_mask] = non_black_replacement_color[c]
        
        return img_tensor


class RandomColorsReplaceBlack():
    """Replaces each black pixel with a unique random color."""
    def __init__(self, p=1.0):
        self.probability = p

    def __call__(self, img_tensor):
        if torch.rand(1).item() > self.probability:
            return img_tensor
        black_pixels_mask = torch.all(img_tensor <= 0.01, dim=0)
        random_colors = torch.rand_like(img_tensor)
        img_tensor[:, black_pixels_mask] = random_colors[:, black_pixels_mask]
        return img_tensor


class RandomColorsReplaceNonBlack():
    """Replaces each non-black pixel with a unique random color."""
    def __init__(self, p=1.0):
        self.probability = p

    def __call__(self, img_tensor):
        if torch.rand(1).item() > self.probability:
            return img_tensor
        non_black_pixels_mask = torch.any(img_tensor > 0.01, dim=0)
        random_colors = torch.rand_like(img_tensor)
        img_tensor[:, non_black_pixels_mask] = random_colors[:, non_black_pixels_mask]
        return img_tensor

    def _normalize_tensor(self, img_tensor):
        """Normalize tensor to float and scale [0, 1] if not already."""
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0
        return img_tensor


class Identity():
    """Replaces each non-black pixel with a unique random color."""
    def __init__(self, p=1.0):
        self.probability = p

    def __call__(self, img_tensor):
        if torch.rand(1).item() > self.probability:
            return img_tensor
        return img_tensor

    def _normalize_tensor(self, img_tensor):
        """Normalize tensor to float and scale [0, 1] if not already."""
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0
        return img_tensor
