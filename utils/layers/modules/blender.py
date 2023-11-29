from typing import Callable

from torch import stack, Tensor


class Blender:
    def __init__(self, blending_function: Callable):
        self.blending_function = blending_function

    def __call__(self, subword_embeddings: Tensor, boundaries: list[int]) -> Tensor:
        blended_tensors: list[Tensor] = []
        for i in range(0, len(boundaries) - 1):
            lower_bound: int = boundaries[i]
            upper_bound: int = boundaries[i + 1]
            combined_word_tensor = self.blending_function(subword_embeddings[lower_bound:upper_bound])
            blended_tensors.append(combined_word_tensor)

        blended_tensor: Tensor = stack(blended_tensors, dim=0)
        return blended_tensor
