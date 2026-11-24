"""Run a script with MLX's buffer cache bounded (the laptop queue's memory guard counts the cache).
Usage: python capped.py <script.py> [args...]   (cache limit: env MLX_CACHE_GB, default 1)"""
import os
import runpy
import sys

import mlx.core as mx

mx.set_cache_limit(int(float(os.environ.get("MLX_CACHE_GB", "1")) * 1e9))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
