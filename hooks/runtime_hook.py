# Runtime hook to fix torch DLL loading on Windows
import os
import sys

def _fix_torch_dll_path():
    """Fix torch DLL loading path for PyInstaller"""
    if sys.platform == 'win32':
        # Get the base path (where the exe is extracted)
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Add torch lib path to DLL search path
        torch_lib_path = os.path.join(base_path, 'torch', 'lib')
        if os.path.exists(torch_lib_path):
            # Add to PATH
            os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
            
            # Use os.add_dll_directory for Python 3.8+
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(torch_lib_path)

_fix_torch_dll_path()
