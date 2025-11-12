# convert_cli.py
# ----------------------
# This is the main Command Line Interface (CLI) entry point for the converter.
#
# It automatically finds and processes all valid sequence folders
# within the provided --input directory.
#
# Example Usage:
# python convert_cli.py --reader idd3d --writer nuscenes --input /path/to/all_idd3d_sequences --output /path/to/my_nuscenes_output
# ----------------------

import argparse
import sys
import logging
import os
import traceback

# --- Import our new Reader and Writer classes ---
from readers import Idd3dReader, BaseReader
from writers import NuScenesWriter, BaseWriter

# This is where we will add new Readers (like ArgoverseReader) in the future
SUPPORTED_READERS: dict[str, type[BaseReader]] = {
    "idd3d": Idd3dReader,
    # "argoverse2": ArgoverseReader 
}

# This is where we will add new Writers (like KittiWriter) in the future
SUPPORTED_WRITERS: dict[str, type[BaseWriter]] = {
    "nuscenes": NuScenesWriter
}

def main():
    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Convert autonomous driving datasets.")
    parser.add_argument("--reader", required=True, choices=SUPPORTED_READERS.keys(),
                        help="The input dataset format.")
    parser.add_argument("--writer", required=True, choices=SUPPORTED_WRITERS.keys(),
                        help="The output dataset format.")
    parser.add_argument("--input", required=True,
                        help="Path to the PARENT directory containing all sequence folders (e.g., /path/to/all_idd3d_sequences).")
    parser.add_argument("--output", required=True,
                        help="Path to the single destination (output) directory.")
    
    args = parser.parse_args()

    # --- 2. Setup Logging (Console Only) ---
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to the console
        ]
    )
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info(f"Starting Conversion Pipeline")
    log.info("=" * 70)
    log.info(f"Reader:     {args.reader}")
    log.info(f"Writer:     {args.writer}")
    log.info(f"Input Dir:  {args.input}")
    log.info(f"Output Dir: {args.output}")
    log.info("-" * 70)

    # --- NEW: Check if input is a valid directory ---
    if not os.path.isdir(args.input):
        log.error(f"Input path is not a valid directory: {args.input}")
        sys.exit(1)

    try:
        # --- 3. Initialize Reader and Writer (ONCE) ---
        log.info(f"Initializing reader: {args.reader}...")
        ReaderClass = SUPPORTED_READERS[args.reader]
        reader = ReaderClass()

        log.info(f"Initializing writer: {args.writer}...")
        WriterClass = SUPPORTED_WRITERS[args.writer]
        writer = WriterClass()

        # --- 4. Find all sequence folders ---
        sequence_folders_to_process = []
        for item_name in sorted(os.listdir(args.input)):
            seq_path = os.path.join(args.input, item_name)
            # Check if it's a directory AND contains annot_data.json (our check for a valid IDD3D seq)
            if os.path.isdir(seq_path) and os.path.exists(os.path.join(seq_path, 'annot_data.json')):
                sequence_folders_to_process.append(seq_path)
            else:
                log.warning(f"Skipping '{item_name}': Not a valid sequence folder (missing annot_data.json or not a directory).")

        if not sequence_folders_to_process:
            log.error(f"No valid sequence folders found in: {args.input}")
            sys.exit(1)
        
        log.info(f"Found {len(sequence_folders_to_process)} sequences to process.")

        # --- 5. Loop and process each sequence ---
        for i, seq_path in enumerate(sequence_folders_to_process):
            log.info("=" * 70)
            log.info(f"Processing sequence {i+1}/{len(sequence_folders_to_process)}: {os.path.basename(seq_path)}")
            log.info("=" * 70)

            # --- 5a. Run the "Read" Step ---
            log.info(f"Reading from source path: {seq_path}")
            intermediate_data = reader.read(seq_path)
            log.info("Successfully read and parsed source data.")

            # --- 5b. Run the "Write" Step ---
            log.info(f"Writing to output path: {args.output}")
            writer.write(intermediate_data, args.output)
        
        log.info("=" * 70)
        log.info("All sequences processed successfully!")
        log.info("=" * 70)
        
    except Exception as e:
        log.error("=" * 70)
        log.error("--- A FATAL ERROR OCCURRED ---")
        log.error(f"Error: {e}")
        log.error("=" * 70)
        log.error(traceback.format_exc()) # Print the full error stack trace
        log.error("Conversion FAILED.")
        sys.exit(1) # Exit with an error code

if __name__ == "__main__":
    main()
