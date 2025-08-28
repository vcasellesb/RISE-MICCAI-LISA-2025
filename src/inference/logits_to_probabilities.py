import torch
import numpy as np


def apply_nonlinearity_to_logits(logits_array: np.ndarray | torch.Tensor) -> torch.Tensor:

    if isinstance(logits_array, np.ndarray):
        logits_array = torch.from_numpy(logits_array)

    with torch.no_grad():
        probabilities = torch.softmax(logits_array.float(), dim=0)

    return probabilities


@torch.inference_mode()
def convert_probabilities_to_segmentation(probabilities_array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    assumes that nonlinearity was already applied!
    probabilities_array has to have shape (c, x, y(, z)) where c is the number of classes
    """

    is_numpy = isinstance(probabilities_array, np.ndarray)

    if not is_numpy:
        probabilities_array = probabilities_array.numpy()

    segmentation = probabilities_array.argmax(0)

    if not is_numpy:
        segmentation = torch.from_numpy(segmentation)

    return segmentation


@torch.inference_mode()
def convert_logits_to_segmentation(
    predicted_logits: np.ndarray | torch.Tensor,
    return_probabilities: bool
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None]:

    input_is_numpy = isinstance(predicted_logits, np.ndarray)

    probabilities = apply_nonlinearity_to_logits(predicted_logits)
    if input_is_numpy:
        probabilities = probabilities.cpu().numpy()

    seg = convert_probabilities_to_segmentation(probabilities)

    if not return_probabilities:
        probabilities = None

    return seg, probabilities
