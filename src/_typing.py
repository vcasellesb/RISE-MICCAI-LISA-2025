from typing import Union, Iterable, List, Tuple
import numpy as np

Number = Union[int, float]
ArrayLike = Union[Iterable[Number], np.ndarray]
ToIterableInt = Union[int, List[int], Tuple[int, ...]]