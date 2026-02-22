#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import download_swedish_electricity_data

if __name__ == '__main__':
    # download 2024-2025 data
    download_swedish_electricity_data('2024-01-01', '2025-12-31')