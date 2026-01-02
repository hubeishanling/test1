# -*- mode: python -*-
# vim: ft=python

import sys
import os

sys.setrecursionlimit(5000)  # required on Windows

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files

# Collect torch completely
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
torch_binaries += collect_dynamic_libs('torch')

# Collect ultralytics
ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = collect_all('ultralytics')

# Collect onnxruntime
onnx_datas, onnx_binaries, onnx_hiddenimports = collect_all('onnxruntime')

# Merge all
all_binaries = torch_binaries + ultralytics_binaries + onnx_binaries
all_datas = torch_datas + ultralytics_datas + onnx_datas
all_hiddenimports = torch_hiddenimports + ultralytics_hiddenimports + onnx_hiddenimports

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=all_binaries + [('pnnx.exe', '.')],
    datas=[
        ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
        ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
        ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling'),
        ('anylabeling/services/auto_labeling/configs/bert/*', 'anylabeling/services/auto_labeling/configs/bert'),
        ('anylabeling/services/auto_labeling/configs/clip/*', 'anylabeling/services/auto_labeling/configs/clip'),
        ('anylabeling/services/auto_labeling/configs/ppocr/*', 'anylabeling/services/auto_labeling/configs/ppocr'),
        ('anylabeling/services/auto_labeling/configs/ram/*', 'anylabeling/services/auto_labeling/configs/ram'),
    ] + all_datas,
    hiddenimports=[
        'anylabeling.views.training',
        'anylabeling.views.training.ultralytics_dialog',
        'anylabeling.views.training.widgets',
        'anylabeling.views.training.widgets.ultralytics_widgets',
        'anylabeling.services.auto_training',
        'anylabeling.services.auto_training.ultralytics',
        'anylabeling.services.auto_training.ultralytics.config',
        'anylabeling.services.auto_training.ultralytics.exporter',
        'anylabeling.services.auto_training.ultralytics.general',
        'anylabeling.services.auto_training.ultralytics.style',
        'anylabeling.services.auto_training.ultralytics.trainer',
        'anylabeling.services.auto_training.ultralytics.utils',
        'anylabeling.services.auto_training.ultralytics.validators',
        'anylabeling.services.auto_training.ultralytics._io',
        'torch._C',
        'torch.utils.data',
    ] + all_hiddenimports,
    hookspath=['hooks'],
    runtime_hooks=['hooks/runtime_hook.py'],
    excludes=[],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='X-AnyLabeling-CPU',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    icon='anylabeling/resources/images/icon.icns',
)

app = BUNDLE(
    exe,
    name='X-AnyLabeling.app',
    icon='anylabeling/resources/images/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
