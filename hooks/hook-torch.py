# PyInstaller hook for torch
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = collect_all('torch')

# Collect all dynamic libraries
binaries += collect_dynamic_libs('torch')
