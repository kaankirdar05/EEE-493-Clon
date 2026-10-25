"""Extended classical detection methods (Part A block detectors, Part B chirp, Part C change-point)."""

import os
import sys

WP4 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "wp4_detection")
if WP4 not in sys.path:
    sys.path.insert(0, WP4)
