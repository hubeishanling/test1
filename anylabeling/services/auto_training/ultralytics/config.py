import os
import multiprocessing

# Ultralytics config path
home_dir = os.path.expanduser("~")
root_dir = os.path.join(home_dir, "xanylabeling_data/trainer/ultralytics")
DATA_PATH = os.path.join(root_dir, "data.yaml")
DATASET_PATH = os.path.join(root_dir, "datasets")
SETTINGS_CONFIG_PATH = os.path.join(root_dir, "settings.json")
DEFAULT_PROJECT_DIR = os.path.join(root_dir, "runs")

# UI configuration
DEFAULT_WINDOW_TITLE = "Ultralytics Training Platforms 🚀"
DEFAULT_WINDOW_SIZE = (1200, 800)  # (w, h)
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)

# Task configuration
TASK_TYPES = ["Detect"]
# TASK_TYPES = ["Classify", "Detect", "OBB", "Segment", "Pose"]
TASK_SHAPE_MAPPINGS = {
    # "Classify": ["flags"],
    "Detect": ["rectangle"],
    # "OBB": ["rotation"],
    # "Segment": ["polygon"],
    # "Pose": ["point"],
}
TASK_LABEL_MAPPINGS = {
    # "Classify": "classify",
    "Detect": "hbb",
    # "OBB": "obb",
    # "Segment": "seg",
    # "Pose": "pose",
}

# Training configuration
MIN_LABELED_IMAGES_THRESHOLD = 20
NUM_WORKERS = multiprocessing.cpu_count()
DEFAULT_TRAINING_CONFIG = {
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "workers": 8,
    "classes": "",
    "single_cls": False,
    "time": 0,
    "patience": 100,
    "close_mosaic": 10,
    "optimizer": "auto",
    "cos_lr": False,
    "amp": True,
    "multi_scale": False,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "dropout": 0.0,
    "fraction": 1.0,
    "rect": False,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 2.0,
    "save_period": -1,
    "val": True,
    "plots": False,
    "save": True,
    "resume": False,
    "cache": False,
}
OPTIMIZER_OPTIONS = [
    "auto",
    "SGD",
    "Adam",
    "AdamW",
    "NAdam",
    "RAdam",
    "RMSProp",
]
TRAINING_STATUS_COLORS = {
    "idle": "#6c757d",
    "training": "#6f42c1",
    "completed": "#28a745",
    "error": "#ffc107",
}
TRAINING_STATUS_TEXTS = {
    "idle": "Ready to train",
    "training": "Training in progress",
    "completed": "Training completed",
    "error": "Training error",
}


# Env Check - Completely lazy, no torch import at module load time
_torch_available = None
_cuda_available = None
_mps_available = None


def _setup_torch_dll_path():
    """Setup DLL path for torch before importing"""
    import sys
    if sys.platform == 'win32' and getattr(sys, 'frozen', False):
        import os
        base_path = sys._MEIPASS
        torch_lib_path = os.path.join(base_path, 'torch', 'lib')
        if os.path.exists(torch_lib_path):
            os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(torch_lib_path)
                except Exception:
                    pass


def is_torch_available() -> bool:
    global _torch_available
    if _torch_available is None:
        try:
            _setup_torch_dll_path()
            import torch
            _torch_available = True
        except Exception:
            _torch_available = False
    return _torch_available


def is_cuda_available() -> bool:
    global _cuda_available
    if _cuda_available is None:
        try:
            if is_torch_available():
                import torch
                _cuda_available = torch.cuda.is_available()
            else:
                _cuda_available = False
        except Exception:
            _cuda_available = False
    return _cuda_available


def is_mps_available() -> bool:
    global _mps_available
    if _mps_available is None:
        try:
            if is_torch_available():
                import torch
                _mps_available = torch.backends.mps.is_available()
            else:
                _mps_available = False
        except Exception:
            _mps_available = False
    return _mps_available


def get_device_options():
    """Get available device options - called at runtime"""
    options = []
    try:
        if is_cuda_available():
            options.append("cuda")
        if is_mps_available():
            options.append("mps")
    except Exception:
        pass
    options.append("cpu")
    return options


# DO NOT call these at module load time!
# Use the functions instead: is_torch_available(), is_cuda_available(), etc.
IS_TORCH_AVAILABLE = None
IS_CUDA_AVAILABLE = None
IS_MPS_AVAILABLE = None
DEVICE_OPTIONS = ["cpu"]  # Default, use get_device_options() at runtime
