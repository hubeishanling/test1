import os
import sys
import threading
from io import StringIO
from typing import Tuple

from PyQt5.QtCore import QObject, pyqtSignal


from .utils import check_package_installed
from .validators import install_packages_with_timeout


class ExportEventRedirector(QObject):
    """Thread-safe export event redirector"""

    export_event_signal = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

    def emit_export_event(self, event_type, data):
        """Safely emit export events from child thread to main thread"""
        self.export_event_signal.emit(event_type, data)


class ExportLogRedirector(QObject):
    """Thread-safe export log redirector"""

    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.log_stream = StringIO()

    def write(self, text):
        """Write text to log stream and emit signal if not empty"""
        if text.strip():
            self.log_signal.emit(text)

    def flush(self):
        """Flush the log stream"""
        pass


def validate_onnx_export_environment():
    required_packages = ["onnx", "onnxslim", "onnxruntime"]
    missing_packages = []
    package_mapping = {
        "onnx": "onnx>=1.12.0,<1.18.0",
        "onnxslim": "onnxslim>=0.1.59",
        "onnxruntime": "onnxruntime",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    try:
        import onnx

        if hasattr(onnx, "__version__"):
            onnx_version = onnx.__version__
            from packaging import version

            if version.parse(onnx_version) >= version.parse("1.18.0"):
                missing_packages.append("onnx>=1.12.0,<1.18.0")
    except:
        pass

    return missing_packages


def validate_openvino_export_environment():
    required_packages = ["openvino"]
    missing_packages = []
    package_mapping = {"openvino": "openvino>=2024.0.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_tensorrt_export_environment():
    required_packages = ["tensorrt"]
    missing_packages = []
    package_mapping = {"tensorrt": "tensorrt>7.0.0,!=10.1.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_coreml_export_environment():
    required_packages = ["coremltools"]
    missing_packages = []
    package_mapping = {"coremltools": "coremltools>=8.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_tensorflow_export_environment():
    required_packages = ["tensorflow"]
    missing_packages = []
    package_mapping = {"tensorflow": "tensorflow>=2.0.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_paddle_export_environment():
    required_packages = ["paddlepaddle", "x2paddle"]
    missing_packages = []
    package_mapping = {
        "paddlepaddle": "paddlepaddle-gpu",
        "x2paddle": "x2paddle",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_mnn_export_environment():
    required_packages = ["MNN"]
    missing_packages = []
    package_mapping = {"MNN": "MNN>=2.9.6"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_ncnn_export_environment():
    """Validate environment for custom NCNN export (Android compatible)."""
    from .ncnn_exporter import validate_ncnn_custom_export_environment
    return validate_ncnn_custom_export_environment()


def validate_imx500_export_environment():
    required_packages = ["imx500-converter", "mct-quantizers"]
    missing_packages = []
    package_mapping = {
        "imx500-converter": "imx500-converter[pt]>=3.16.1",
        "mct-quantizers": "mct-quantizers>=1.6.0",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_rknn_export_environment():
    required_packages = ["rknn-toolkit2"]
    missing_packages = []
    package_mapping = {"rknn-toolkit2": "rknn-toolkit2"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def get_export_validator(export_format):
    validators = {
        "onnx": validate_onnx_export_environment,
        "openvino": validate_openvino_export_environment,
        "engine": validate_tensorrt_export_environment,
        "coreml": validate_coreml_export_environment,
        "saved_model": validate_tensorflow_export_environment,
        "pb": validate_tensorflow_export_environment,
        "tflite": validate_tensorflow_export_environment,
        "edgetpu": validate_tensorflow_export_environment,
        "tfjs": validate_tensorflow_export_environment,
        "paddle": validate_paddle_export_environment,
        "mnn": validate_mnn_export_environment,
        "ncnn": validate_ncnn_export_environment,
        "imx": validate_imx500_export_environment,
        "rknn": validate_rknn_export_environment,
        "torchscript": lambda: [],
    }
    return validators.get(export_format, lambda: [])


class ExportManager:
    def __init__(self):
        self.is_exporting = False
        self.callbacks = []
        self.export_thread = None

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Error in export callback: {e}")

    def start_export(
        self, project_path: str, export_format: str = "onnx"
    ) -> Tuple[bool, str]:
        if self.is_exporting:
            return False, "Export already in progress"

        weights_path = os.path.join(project_path, "weights", "best.pt")
        if not os.path.exists(weights_path):
            return False, f"Model weights not found at: {weights_path}"

        self.is_exporting = True
        self.export_thread = threading.Thread(
            target=self._export_worker, args=(weights_path, export_format)
        )
        self.export_thread.start()
        return True, "Export started successfully"

    def start_export_file(
        self, weights_path: str, export_format: str = "onnx", output_dir: str = None
    ) -> Tuple[bool, str]:
        """
        Export a model file directly without requiring a project structure.

        Args:
            weights_path: Path to the .pt model file
            export_format: Target export format
            output_dir: Output directory (default: same as weights file)

        Returns:
            Tuple of (success, message)
        """
        if self.is_exporting:
            return False, "Export already in progress"

        if not os.path.exists(weights_path):
            return False, f"Model file not found: {weights_path}"

        if not weights_path.endswith(".pt"):
            return False, "Model file must be a .pt file"

        if output_dir is None:
            output_dir = os.path.dirname(weights_path)

        self.is_exporting = True
        self.export_thread = threading.Thread(
            target=self._export_worker,
            args=(weights_path, export_format, output_dir),
        )
        self.export_thread.start()
        return True, "Export started successfully"

    def _export_worker(
        self, weights_path: str, export_format: str, output_dir: str = None
    ):
        try:
            self.notify_callbacks(
                "export_started",
                {"weights_path": weights_path, "format": export_format},
            )
            self.notify_callbacks(
                "export_log", {"message": "Checking export environment..."}
            )
            missing_packages = get_export_validator(export_format)()
            if missing_packages:
                self.notify_callbacks(
                    "export_log",
                    {
                        "message": f"Missing required packages: {', '.join(missing_packages)}"
                    },
                )
                self.notify_callbacks(
                    "export_log",
                    {"message": "Attempting to install missing packages..."},
                )
                success, stdout, stderr = install_packages_with_timeout(
                    missing_packages, timeout=30
                )
                if not success:
                    error_msg = f"Failed to install required packages: {', '.join(missing_packages)}. Please manually install these packages and restart the application."
                    self.notify_callbacks("export_error", {"error": error_msg})
                    return
                self.notify_callbacks(
                    "export_log",
                    {"message": "Required packages installed successfully"},
                )
            else:
                self.notify_callbacks(
                    "export_log",
                    {"message": "All required packages are available"},
                )

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            log_redirector = ExportLogRedirector()
            sys.stdout = log_redirector
            sys.stderr = log_redirector

            try:
                # Use custom NCNN exporter for Android compatibility
                if export_format == "ncnn":
                    self._export_ncnn_android(weights_path, output_dir)
                else:
                    self._export_standard(weights_path, export_format, output_dir)

            except ImportError as e:
                self.notify_callbacks(
                    "export_error",
                    {"error": f"Failed to import required module: {str(e)}"},
                )
            except Exception as e:
                self.notify_callbacks(
                    "export_error", {"error": f"Export failed: {str(e)}"}
                )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        except Exception as e:
            self.notify_callbacks(
                "export_error",
                {"error": f"Unexpected error during export: {str(e)}"},
            )
        finally:
            self.is_exporting = False

    def _export_ncnn_android(self, weights_path: str, output_dir: str = None):
        """Export model to NCNN format compatible with Android (RichAuto format)."""
        from .ncnn_exporter import export_ncnn_for_android

        def log_callback(message: str):
            self.notify_callbacks("export_log", {"message": message})

        self.notify_callbacks(
            "export_log",
            {"message": "Starting custom NCNN export for Android..."},
        )

        if output_dir is None:
            output_dir = os.path.dirname(weights_path)

        param_path, bin_path = export_ncnn_for_android(
            weights_path, output_dir, log_callback
        )

        self.notify_callbacks(
            "export_completed",
            {
                "exported_path": param_path,
                "format": "ncnn",
                "additional_files": [bin_path],
            },
        )

    def _export_standard(self, weights_path: str, export_format: str, output_dir: str = None):
        """Standard export using ultralytics."""
        from ultralytics import YOLO

        self.notify_callbacks(
            "export_log",
            {"message": f"Loading model from {weights_path}"},
        )
        model = YOLO(weights_path)

        self.notify_callbacks(
            "export_log",
            {"message": f"Starting export to {export_format} format..."},
        )

        # Build export kwargs
        export_kwargs = {"format": export_format}
        if output_dir:
            # For some formats, we need to handle output directory differently
            pass  # ultralytics handles output location automatically

        results = model.export(**export_kwargs)

        exported_path = results if isinstance(results, str) else str(results)
        if not exported_path:
            weights_dir = output_dir or os.path.dirname(weights_path)
            model_name = os.path.splitext(os.path.basename(weights_path))[0]
            exported_path = os.path.join(
                weights_dir, f"{model_name}.{export_format}"
            )

        if os.path.exists(exported_path):
            self.notify_callbacks(
                "export_completed",
                {"exported_path": exported_path, "format": export_format},
            )
        else:
            possible_path = weights_path.replace(".pt", f".{export_format}")
            if os.path.exists(possible_path):
                self.notify_callbacks(
                    "export_completed",
                    {"exported_path": possible_path, "format": export_format},
                )
            else:
                self.notify_callbacks(
                    "export_error",
                    {"error": "Export completed but output file not found"},
                )

    def stop_export(self) -> bool:
        if not self.is_exporting:
            return False

        self.is_exporting = False
        if self.export_thread and self.export_thread.is_alive():
            self.export_thread.join(timeout=5)

        self.notify_callbacks("export_stopped", {})
        return True


_export_manager = None


def get_export_manager() -> ExportManager:
    global _export_manager
    if _export_manager is None:
        _export_manager = ExportManager()
    return _export_manager


def export_model(
    project_path: str, export_format: str = "onnx"
) -> Tuple[bool, str]:
    manager = get_export_manager()
    return manager.start_export(project_path, export_format)
