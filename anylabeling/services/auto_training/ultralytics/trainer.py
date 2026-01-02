import os
import signal
import shutil
import subprocess
import sys
import time
import threading
from io import StringIO
from typing import Dict, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from .config import SETTINGS_CONFIG_PATH


class TrainingEventRedirector(QObject):
    """Thread-safe training event redirector"""

    training_event_signal = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

    def emit_training_event(self, event_type, data):
        """Safely emit training events from child thread to main thread"""
        self.training_event_signal.emit(event_type, data)


class TrainingLogRedirector(QObject):
    """Thread-safe training log redirector"""

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


class TrainingManager:
    def __init__(self):
        self.training_process = None
        self.is_training = False
        self.callbacks = []
        self.total_epochs = 100
        self.stop_event = threading.Event()

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception:
                pass

    def start_training(self, train_args: Dict) -> Tuple[bool, str]:
        if self.is_training:
            return False, "Training is already in progress"

        try:
            self.total_epochs = train_args.get("epochs", 100)
            self.stop_event.clear()

            # Store train_args for the training thread
            self._train_args = train_args.copy()

            def run_training():
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    
                    from ultralytics import YOLO
                    
                    self.is_training = True
                    self.notify_callbacks(
                        "training_started", {"total_epochs": self.total_epochs}
                    )

                    model_path = self._train_args.pop("model")
                    model = YOLO(model_path)
                    
                    # Set training args
                    self._train_args['verbose'] = True
                    self._train_args['show'] = False
                    
                    # Add callback to check for stop event
                    def on_train_epoch_end(trainer):
                        if self.stop_event.is_set():
                            raise KeyboardInterrupt("Training stopped by user")
                        # Log progress
                        epoch = trainer.epoch + 1
                        self.notify_callbacks(
                            "training_log",
                            {"message": f"Epoch {epoch}/{self.total_epochs} completed"}
                        )
                    
                    model.add_callback("on_train_epoch_end", on_train_epoch_end)
                    
                    # Redirect stdout to capture training logs
                    import io
                    import contextlib
                    
                    class LogCapture(io.StringIO):
                        def __init__(self, callback):
                            super().__init__()
                            self.callback = callback
                        
                        def write(self, text):
                            if text.strip():
                                self.callback("training_log", {"message": text.strip()})
                            return super().write(text)
                    
                    log_capture = LogCapture(self.notify_callbacks)
                    
                    try:
                        with contextlib.redirect_stdout(log_capture):
                            results = model.train(**self._train_args)
                        
                        self.is_training = False
                        self.notify_callbacks(
                            "training_completed",
                            {"results": "Training completed successfully"},
                        )
                    except KeyboardInterrupt:
                        self.is_training = False
                        self.notify_callbacks("training_stopped", {})
                    except Exception as e:
                        self.is_training = False
                        self.notify_callbacks(
                            "training_error",
                            {"error": f"Training error: {str(e)}"}
                        )

                except Exception as e:
                    self.is_training = False
                    self.notify_callbacks("training_error", {"error": str(e)})

            def save_settings_config():
                save_path = os.path.join(
                    self._train_args.get("project", ""), 
                    self._train_args.get("name", "")
                )
                save_file = os.path.join(save_path, "settings.json")

                # Wait for directory to be created
                timeout = 60
                start_time = time.time()
                while not os.path.exists(save_path):
                    if time.time() - start_time > timeout:
                        return
                    time.sleep(1)

                try:
                    shutil.copy2(SETTINGS_CONFIG_PATH, save_file)
                except Exception:
                    pass

            training_thread = threading.Thread(target=run_training)
            training_thread.daemon = True
            training_thread.start()

            config_thread = threading.Thread(target=save_settings_config)
            config_thread.daemon = True
            config_thread.start()

            return True, "Training started successfully"

        except ImportError:
            return (
                False,
                "Ultralytics is not installed. Please install it with: pip install ultralytics",
            )
        except Exception as e:
            return False, f"Failed to start training: {str(e)}"

    def stop_training(self) -> bool:
        if not self.is_training:
            return False

        try:
            self.stop_event.set()
            return True
        except Exception:
            return False


_training_manager = TrainingManager()


def get_training_manager() -> TrainingManager:
    return _training_manager
