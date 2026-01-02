"""
Custom NCNN exporter for YOLOv8 models.

This module provides a custom NCNN export pipeline that produces models
compatible with Android NCNN inference (RichAuto format).

The export process:
1. Export PyTorch model to TorchScript
2. Convert TorchScript to PNNX format
3. Modify the generated pnnx.py to customize output format
4. Re-export modified model to TorchScript
5. Convert to final NCNN format

Output format: [1, 8400, num_channels] where num_channels = 64 + num_classes
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Optional, Tuple


def get_pnnx_path() -> str:
    """Get the path to pnnx executable, handling PyInstaller frozen environment."""
    import sys
    import shutil as sh
    
    # If running in PyInstaller bundle, check the bundle directory first
    if getattr(sys, 'frozen', False):
        # pnnx.exe should be in the same directory as the exe
        bundle_dir = sys._MEIPASS
        pnnx_in_bundle = os.path.join(bundle_dir, 'pnnx.exe')
        if os.path.exists(pnnx_in_bundle):
            return pnnx_in_bundle
    
    # Try to find pnnx in PATH
    pnnx_path = sh.which('pnnx')
    if pnnx_path:
        return pnnx_path
    
    # Try pnnx.exe specifically on Windows
    if sys.platform == 'win32':
        pnnx_path = sh.which('pnnx.exe')
        if pnnx_path:
            return pnnx_path
    
    return None


def check_pnnx_available() -> Tuple[bool, str]:
    """Check if pnnx command is available."""
    pnnx_path = get_pnnx_path()
    if not pnnx_path:
        return False, "pnnx command not found. Please install pnnx first."
    
    try:
        result = subprocess.run(
            [pnnx_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "pnnx command timed out."
    except Exception as e:
        return False, f"Error checking pnnx: {str(e)}"


def validate_ncnn_custom_export_environment() -> list:
    """Validate environment for custom NCNN export."""
    missing_packages = []

    # Check pnnx
    pnnx_ok, pnnx_error = check_pnnx_available()
    if not pnnx_ok:
        missing_packages.append("pnnx")

    return missing_packages


def get_num_classes_from_model(weights_path: str) -> int:
    """Get number of classes from YOLO model."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return len(model.names)
    except Exception as e:
        raise RuntimeError(f"Failed to get num_classes from model: {e}")


def export_to_torchscript(
    weights_path: str,
    output_dir: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Export YOLO model to TorchScript format."""
    from ultralytics import YOLO

    if log_callback:
        log_callback(f"Loading model from {weights_path}")

    model = YOLO(weights_path)

    if log_callback:
        log_callback("Exporting to TorchScript format...")

    # Export to torchscript
    ts_path = model.export(format="torchscript")

    if not ts_path or not os.path.exists(ts_path):
        # Try to find the exported file
        base_name = os.path.splitext(os.path.basename(weights_path))[0]
        ts_path = os.path.join(os.path.dirname(weights_path), f"{base_name}.torchscript")

    if not os.path.exists(ts_path):
        raise RuntimeError("TorchScript export failed: output file not found")

    if log_callback:
        log_callback(f"TorchScript exported to: {ts_path}")

    return ts_path


def run_pnnx_convert(
    torchscript_path: str,
    work_dir: str,
    input_shapes: list = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Run pnnx to convert TorchScript to PNNX format."""
    if log_callback:
        log_callback(f"Running pnnx on {torchscript_path}")

    pnnx_path = get_pnnx_path()
    if not pnnx_path:
        raise RuntimeError("pnnx not found. Please ensure pnnx is installed.")

    cmd = [pnnx_path, torchscript_path]

    if input_shapes:
        for shape in input_shapes:
            cmd.append(f"inputshape={shape}")

    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if log_callback and result.stdout:
            log_callback(result.stdout)

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            raise RuntimeError(f"pnnx conversion failed: {error_msg}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("pnnx conversion timed out (>5 minutes)")

    # Find generated pnnx.py file
    base_name = os.path.splitext(os.path.basename(torchscript_path))[0]
    pnnx_py_path = os.path.join(work_dir, f"{base_name}_pnnx.py")

    if not os.path.exists(pnnx_py_path):
        # Try alternative naming
        for f in os.listdir(work_dir):
            if f.endswith("_pnnx.py"):
                pnnx_py_path = os.path.join(work_dir, f)
                break

    if not os.path.exists(pnnx_py_path):
        raise RuntimeError("pnnx conversion failed: _pnnx.py file not found")

    if log_callback:
        log_callback(f"PNNX file generated: {pnnx_py_path}")

    return pnnx_py_path


def fix_pnnx_py_paths(
    pnnx_py_path: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Fix Windows path issues in pnnx.py file.

    PNNX generates paths with backslashes on Windows, which causes
    'unicodeescape' codec errors when Python tries to parse them.
    This function converts all backslashes in string literals to forward slashes.
    """
    if log_callback:
        log_callback("Fixing Windows path issues in pnnx.py...")

    with open(pnnx_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple approach: replace all backslashes with forward slashes in the file
    # This is safe because:
    # 1. Python code doesn't use backslashes except in strings
    # 2. Forward slashes work on Windows for file paths
    # 3. The pnnx.py file only contains model code and paths

    # Replace backslashes in double-quoted strings
    def fix_double_quoted(match):
        full_match = match.group(0)
        return full_match.replace("\\", "/")

    # Replace backslashes in single-quoted strings
    def fix_single_quoted(match):
        full_match = match.group(0)
        return full_match.replace("\\", "/")

    # Fix double-quoted strings containing backslashes
    content = re.sub(r'"[^"]*\\[^"]*"', fix_double_quoted, content)

    # Fix single-quoted strings containing backslashes
    content = re.sub(r"'[^']*\\[^']*'", fix_single_quoted, content)

    with open(pnnx_py_path, "w", encoding="utf-8") as f:
        f.write(content)

    if log_callback:
        log_callback("Fixed Windows path issues")


def modify_pnnx_py_for_android(
    pnnx_py_path: str,
    num_classes: int,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Modify the generated pnnx.py file to output Android-compatible format.

    Changes the output from post-processed detection results to raw tensor
    format: [1, 8400, num_channels] where num_channels = 64 + num_classes
    
    Based on model_converter.py reference implementation.
    """
    if log_callback:
        log_callback(f"Modifying {pnnx_py_path} for Android compatibility...")

    with open(pnnx_py_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Calculate expected channel count (64 for bbox regression + num_classes)
    num_channels = 64 + num_classes

    if log_callback:
        log_callback(
            f"Model has {num_classes} classes, output channels: {num_channels}"
        )

    # Find the line with torch.cat(..., dim=2)
    cat_line_idx = -1
    cat_line = None
    cat_var = None
    cat_input_vars = []

    for i, line in enumerate(lines):
        if "torch.cat" in line and "dim=2" in line and "= torch.cat" in line:
            cat_line_idx = i
            cat_line = line
            # Extract the output variable name and input variables
            match = re.match(r"\s*(v_\d+)\s*=\s*torch\.cat\(\((.*?)\),\s*dim=2\)", line)
            if match:
                cat_var = match.group(1)
                cat_input_vars = [v.strip() for v in match.group(2).split(",")]
            if log_callback:
                log_callback(f"Found torch.cat at line {i + 1}: {line.strip()}")
                log_callback(f"Cat output: {cat_var}, inputs: {cat_input_vars}")
            break

    if cat_line_idx == -1:
        raise RuntimeError(
            "Could not find torch.cat with dim=2 in pnnx.py. "
            "The model structure may not be supported."
        )

    # Get indentation from cat line
    indent = cat_line[: len(cat_line) - len(cat_line.lstrip())]

    # Search backwards to find the view/reshape operations for each input variable
    # Look for patterns like: v_165 = v_142.view(1, 144, 6400) or v_165 = v_142.reshape(1, 145, 6400)
    view_info = {}
    
    for j in range(cat_line_idx - 1, max(0, cat_line_idx - 30), -1):
        line_content = lines[j]
        for var_name in cat_input_vars:
            # Support both .view() and .reshape() - newer PyTorch versions use reshape
            if var_name not in view_info and var_name in line_content and (".view(" in line_content or ".reshape(" in line_content):
                # Match: v_XXX = v_YYY.view(1, NUM, NUM) or v_XXX = v_YYY.reshape(1, NUM, NUM)
                match = re.search(
                    r"(v_\d+)\s*=\s*(v_\d+)\.(view|reshape)\(\s*1\s*,\s*(\d+)\s*,\s*(-?\d+)\s*\)",
                    line_content,
                )
                if match and match.group(1) == var_name:
                    view_info[var_name] = {
                        "line_idx": j,
                        "out_var": match.group(1),
                        "in_var": match.group(2),
                        "channels": match.group(4),  # Keep as string for now
                        "grid_size": match.group(5),
                        "original_line": line_content,
                    }
                    if log_callback:
                        log_callback(
                            f"Found {match.group(3)} for {var_name} at line {j + 1}: "
                            f"dim={match.group(4)}"
                        )

    if log_callback:
        log_callback(f"Found {len(view_info)} view operations out of {len(cat_input_vars)} expected")

    # Find the return statement after cat
    return_line_idx = -1
    for j in range(cat_line_idx + 1, min(len(lines), cat_line_idx + 30)):
        if lines[j].strip().startswith("return"):
            return_line_idx = j
            break

    if return_line_idx == -1:
        raise RuntimeError("Could not find return statement after torch.cat")

    if len(view_info) >= len(cat_input_vars):
        # We found all view operations - use the standard approach
        if log_callback:
            log_callback("Using standard modification approach...")
        
        # Build new lines
        new_lines = []

        # Add all lines before the first view operation
        start_idx = min(v["line_idx"] for v in view_info.values())
        new_lines.extend(lines[:start_idx])

        # Add modified view operations with transpose
        # Use the channel dimension from each view operation
        for var_name in cat_input_vars:
            v = view_info[var_name]
            new_line = (
                f"{indent}{v['out_var']} = "
                f"{v['in_var']}.view(1, {v['channels']}, -1).transpose(1, 2)\n"
            )
            new_lines.append(new_line)

        # Add modified cat operation (dim=1 instead of dim=2)
        new_lines.append(f"{indent}{cat_var} = torch.cat(({', '.join(cat_input_vars)}), dim=1)\n")

        # Add return statement
        new_lines.append(f"{indent}return {cat_var}\n")

        # Add remaining lines after return (class definitions, export functions, etc.)
        new_lines.extend(lines[return_line_idx + 1:])

    else:
        # Fallback: Try to find view operations by looking for .view(1, pattern
        if log_callback:
            log_callback("Using fallback modification approach...")
        
        # Look for any .view(1, or .reshape(1, operations near the cat line
        view_lines_found = []
        for j in range(max(0, cat_line_idx - 20), cat_line_idx):
            line_content = lines[j]
            if (".view(1," in line_content or ".reshape(1," in line_content) and "v_" in line_content and "=" in line_content:
                match = re.search(
                    r"(v_\d+)\s*=\s*(v_\d+)\.(view|reshape)\(\s*1\s*,\s*(\d+)\s*,",
                    line_content,
                )
                if match:
                    view_lines_found.append({
                        "line_idx": j,
                        "out_var": match.group(1),
                        "in_var": match.group(2),
                        "channels": match.group(4),
                    })
                    if log_callback:
                        log_callback(f"Found {match.group(3)} at line {j + 1}: {match.group(1)} dim={match.group(4)}")
        
        if len(view_lines_found) >= 3:
            # Use the last 3 view operations (they should be the ones feeding into cat)
            view_lines_found = view_lines_found[-3:]
            
            # Build new lines
            new_lines = []
            start_idx = view_lines_found[0]["line_idx"]
            new_lines.extend(lines[:start_idx])
            
            # Add modified view operations
            for vl in view_lines_found:
                new_line = (
                    f"{indent}{vl['out_var']} = "
                    f"{vl['in_var']}.view(1, {vl['channels']}, -1).transpose(1, 2)\n"
                )
                new_lines.append(new_line)
            
            # Add modified cat operation
            out_vars = [vl['out_var'] for vl in view_lines_found]
            new_lines.append(f"{indent}{cat_var} = torch.cat(({', '.join(out_vars)}), dim=1)\n")
            new_lines.append(f"{indent}return {cat_var}\n")
            
            # Add remaining lines after return
            new_lines.extend(lines[return_line_idx + 1:])
        else:
            # Last resort: just modify the cat line directly
            if log_callback:
                log_callback("Using minimal modification approach (transpose inputs)...")
            
            new_lines = lines[:cat_line_idx]
            
            # Add transpose for each input variable
            for var in cat_input_vars:
                new_lines.append(f"{indent}{var} = {var}.transpose(1, 2)\n")
            
            # Modified cat with dim=1
            new_lines.append(f"{indent}{cat_var} = torch.cat(({', '.join(cat_input_vars)}), dim=1)\n")
            new_lines.append(f"{indent}return {cat_var}\n")
            
            # Add remaining lines after return
            new_lines.extend(lines[return_line_idx + 1:])

    # Write modified file
    with open(pnnx_py_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    if log_callback:
        log_callback("Successfully modified pnnx.py for Android format")

    return pnnx_py_path


def export_modified_torchscript(
    pnnx_py_path: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Export the modified pnnx.py back to TorchScript."""
    if log_callback:
        log_callback("Exporting modified model to TorchScript...")

    work_dir = os.path.dirname(pnnx_py_path)
    module_name = os.path.splitext(os.path.basename(pnnx_py_path))[0]

    # Add work_dir to Python path
    if work_dir not in sys.path:
        sys.path.insert(0, work_dir)

    try:
        # Import and export
        import importlib
        module = importlib.import_module(module_name)

        # Reload in case it was previously imported
        importlib.reload(module)

        # Call export_torchscript function
        if hasattr(module, "export_torchscript"):
            module.export_torchscript()
        else:
            raise RuntimeError(
                f"Module {module_name} does not have export_torchscript function"
            )

    finally:
        # Remove from path
        if work_dir in sys.path:
            sys.path.remove(work_dir)

    # Find the exported .pt file
    expected_pt = os.path.join(work_dir, f"{module_name}.pt")

    if not os.path.exists(expected_pt):
        # Try to find any .pt file created
        for f in os.listdir(work_dir):
            if f.endswith(".pt") and module_name in f:
                expected_pt = os.path.join(work_dir, f)
                break

    if not os.path.exists(expected_pt):
        raise RuntimeError("Failed to export modified TorchScript")

    if log_callback:
        log_callback(f"Modified TorchScript exported: {expected_pt}")

    return expected_pt


def convert_to_final_ncnn(
    modified_pt_path: str,
    output_dir: str,
    model_name: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str]:
    """Convert modified TorchScript to final NCNN format."""
    if log_callback:
        log_callback("Converting to final NCNN format...")

    pnnx_path = get_pnnx_path()
    if not pnnx_path:
        raise RuntimeError("pnnx not found. Please ensure pnnx is installed.")

    work_dir = os.path.dirname(modified_pt_path)

    cmd = [
        pnnx_path,
        modified_pt_path,
        "inputshape=[1,3,640,640]",
        "inputshape2=[1,3,320,320]",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if log_callback and result.stdout:
            log_callback(result.stdout)

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            raise RuntimeError(f"Final pnnx conversion failed: {error_msg}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Final pnnx conversion timed out")

    # Find generated ncnn files
    base_name = os.path.splitext(os.path.basename(modified_pt_path))[0]

    param_file = None
    bin_file = None

    for f in os.listdir(work_dir):
        if f.endswith(".ncnn.param"):
            param_file = os.path.join(work_dir, f)
        elif f.endswith(".ncnn.bin"):
            bin_file = os.path.join(work_dir, f)

    if not param_file or not bin_file:
        raise RuntimeError("NCNN conversion failed: .param or .bin file not found")

    # Copy to output directory with proper names
    output_param = os.path.join(output_dir, f"{model_name}.ncnn.param")
    output_bin = os.path.join(output_dir, f"{model_name}.ncnn.bin")

    shutil.copy2(param_file, output_param)
    shutil.copy2(bin_file, output_bin)

    if log_callback:
        log_callback(f"NCNN model exported to:")
        log_callback(f"  - {output_param}")
        log_callback(f"  - {output_bin}")

    return output_param, output_bin


def export_ncnn_for_android(
    weights_path: str,
    output_dir: str = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str]:
    """
    Export YOLO model to NCNN format compatible with Android (RichAuto format).

    Args:
        weights_path: Path to the .pt weights file
        output_dir: Output directory for NCNN files (default: same as weights)
        log_callback: Optional callback for logging progress

    Returns:
        Tuple of (param_path, bin_path)
    """
    if output_dir is None:
        output_dir = os.path.dirname(weights_path)

    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(weights_path))[0]

    # Create temporary working directory
    temp_dir = tempfile.mkdtemp(prefix="ncnn_export_")

    try:
        if log_callback:
            log_callback(f"Working directory: {temp_dir}")

        # Step 1: Get number of classes
        num_classes = get_num_classes_from_model(weights_path)
        if log_callback:
            log_callback(f"Detected {num_classes} classes in model")

        # Step 2: Export to TorchScript
        ts_path = export_to_torchscript(weights_path, temp_dir, log_callback)

        # Copy torchscript to temp dir for pnnx
        ts_temp_path = os.path.join(temp_dir, os.path.basename(ts_path))
        if ts_path != ts_temp_path:
            shutil.copy2(ts_path, ts_temp_path)
            ts_path = ts_temp_path

        # Step 3: Run pnnx to get pnnx.py
        pnnx_py_path = run_pnnx_convert(ts_path, temp_dir, log_callback=log_callback)

        # Save a copy of original pnnx.py for debugging
        debug_pnnx_path = os.path.join(output_dir, f"{model_name}_pnnx_original.py")
        shutil.copy2(pnnx_py_path, debug_pnnx_path)
        if log_callback:
            log_callback(f"Saved original pnnx.py to: {debug_pnnx_path}")

        # Step 4: Modify pnnx.py for Android format
        modify_pnnx_py_for_android(pnnx_py_path, num_classes, log_callback)

        # Step 4.5: Fix Windows path issues in pnnx.py
        fix_pnnx_py_paths(pnnx_py_path, log_callback)

        # Save modified pnnx.py for debugging
        debug_modified_path = os.path.join(output_dir, f"{model_name}_pnnx_modified.py")
        shutil.copy2(pnnx_py_path, debug_modified_path)
        if log_callback:
            log_callback(f"Saved modified pnnx.py to: {debug_modified_path}")

        # Step 5: Export modified model to TorchScript
        modified_pt_path = export_modified_torchscript(pnnx_py_path, log_callback)

        # Step 6: Convert to final NCNN format
        param_path, bin_path = convert_to_final_ncnn(
            modified_pt_path, output_dir, model_name, log_callback
        )

        # Clean up temp directory on success
        shutil.rmtree(temp_dir, ignore_errors=True)

        return param_path, bin_path

    except Exception as e:
        if log_callback:
            log_callback(f"Error during export. Temp files preserved at: {temp_dir}")
        raise
