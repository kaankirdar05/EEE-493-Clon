"""Newly added classical detection methods (eigenvalue family, TKEO, stochastic resonance).

Importing this package puts ../wp4_detection on sys.path so `wp4` (the verified
baseline framework) can be imported unchanged.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WP4 = os.path.join(ROOT, "wp4_detection")
if WP4 not in sys.path:
    sys.path.insert(0, WP4)
