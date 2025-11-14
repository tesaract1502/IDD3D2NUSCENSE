# utils.py
# ----------------------
# Contains helper functions for JSON manipulation and file operations.
# Token management has been moved to writer-specific implementations.
# ----------------------

import os
import json
import logging
import threading

log = logging.getLogger(__name__)

# This lock is used by JSON file operations to prevent race conditions
json_file_lock = threading.Lock()


def append_to_json_list(file_path, new_data_list):
    """
    Safely appends a list of new data to an existing JSON file
    that is expected to contain a list.
    
    Args:
        file_path: Path to the JSON file
        new_data_list: List of new entries to append
    """
    if not new_data_list:
        log.info(f"No new data to append to {os.path.basename(file_path)}")
        return

    with json_file_lock:
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        log.warning(f"{file_path} is not a list. Overwriting.")
                        existing_data = []
            except json.JSONDecodeError:
                log.warning(f"{file_path} is corrupted. Overwriting.")
                existing_data = []
        
        final_data = existing_data + new_data_list
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            log.info(f"Appended {len(new_data_list)} items to {os.path.basename(file_path)}. Total: {len(final_data)}")
        except Exception as e:
            log.error(f"FATAL: Could not write to {file_path}: {e}")
            raise


def merge_and_overwrite_json_list(file_path, new_entries, key_field):
    """
    Safely merges a list of new entries into a JSON file,
    using 'key_field' to check for uniqueness.
    
    This reads the file, updates a dictionary, and overwrites the file.
    
    Args:
        file_path: Path to the JSON file
        new_entries: List of new entries to merge
        key_field: Field name to use as unique key for merging
    """
    with json_file_lock:
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_list = json.load(f)
                    if isinstance(existing_list, list):
                        for entry in existing_list:
                            if key_field in entry:
                                existing_data[entry[key_field]] = entry
                    else:
                        log.warning(f"{file_path} is not a list. Overwriting.")
            except json.JSONDecodeError:
                log.warning(f"{file_path} is corrupted. Overwriting.")
        
        # Add new entries, overwriting duplicates
        for entry in new_entries:
            if key_field in entry:
                existing_data[entry[key_field]] = entry
        
        final_list = list(existing_data.values())
        
        try:
            with open(file_path, 'w') as f:
                json.dump(final_list, f, indent=2)
            log.info(f"Merged/overwrote {os.path.basename(file_path)}. Total items: {len(final_list)}")
        except Exception as e:
            log.error(f"FATAL: Could not write to {file_path}: {e}")
            raise


def load_json_safely(file_path, default=None):
    """
    Safely loads a JSON file with error handling.
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return if file doesn't exist or is invalid
        
    Returns:
        Parsed JSON data or default value
    """
    if default is None:
        default = []
        
    if not os.path.exists(file_path):
        return default
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning(f"Could not parse {file_path}. Returning default.")
        return default
    except Exception as e:
        log.error(f"Error reading {file_path}: {e}")
        return default


def save_json_safely(file_path, data, indent=2):
    """
    Safely saves data to a JSON file with error handling.
    
    Args:
        file_path: Path to the JSON file
        data: Data to save
        indent: JSON indentation level
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        log.debug(f"Saved {os.path.basename(file_path)}")
    except Exception as e:
        log.error(f"FATAL: Could not write to {file_path}: {e}")
        raise


def ensure_directory(directory_path):
    """
    Ensures a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        log.debug(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        log.error(f"Could not create directory {directory_path}: {e}")
        raise
