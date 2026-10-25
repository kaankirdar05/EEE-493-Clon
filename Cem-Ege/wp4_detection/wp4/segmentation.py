"""Job 4.3 - turn a detection mask into one box per transmission.

Works on the output of ANY detector (energy, CA-CFAR, ...): clean the mask with
morphology, label connected regions, and take each region's bounding box.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def segment(mask: np.ndarray, close: tuple[int, int] = (17, 5), open_: tuple[int, int] = (3, 3),
            min_cells: int = 120) -> list[tuple[int, int, int, int]]:
    """Mask (T, F) -> list of boxes (t0, t1, f0, f1), half-open cell ranges.

    close:     closing kernel (time, freq) to bridge small holes inside a signal
    open_:     opening kernel to delete isolated false-alarm specks
    min_cells: drop regions smaller than this many cells. 120 sits just below the
               smallest TorchSig emitter (5 % of 512 frames x fs/100 = ~130 cells).
               300 gave higher F1 on dev scenes 0-99 (0.59 vs 0.53, same recall)
               but would delete the shortest legitimate bursts.

    Defaults tuned on dev scenes 0-99 of the impaired set; report on 100-299.
    """
    m = mask
    if close:
        m = ndimage.binary_closing(m, structure=np.ones(close, bool))
    if open_:
        m = ndimage.binary_opening(m, structure=np.ones(open_, bool))
    labels, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    sizes = ndimage.sum(m, labels, index=np.arange(1, n + 1))
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labels)):
        if sl is not None and sizes[i] >= min_cells:
            boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return boxes
