# convert_cli.py
# ----------------------
# This is the main Command Line Interface (CLI) entry point for the converter.
#
# It parses user arguments (like input/output paths) and then
# runs the selected Reader and Writer to perform the conversion.
#
# Example Usage:
# python convert_cli.py --reader idd3d --writer nuscenes --input /path/to/idd3d_seq10 --output /path/to/my_nuscenes_output
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
                        help="Path to the source sequence directory (e.g., /path/to/idd3d_seq10).")
    parser.add_argument("--output", required=True,
                        help="Path to the destination (output) directory (e.g., /path/to/my_nuscenes_output).")
    
    args = parser.parse_args()

    # --- 2. Setup Logging (Console Only) ---
    # --- MODIFIED: Removed FileHandler ---
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
    log.info(f"Reader:   {args.reader}")
    log.info(f"Writer:   {args.writer}")
    log.info(f"Input:    {args.input}")
    log.info(f"Output:   {args.output}")
    log.info("-" * 70)

    try:
        # --- 3. Initialize Reader ---
        log.info(f"Initializing reader: {args.reader}...")
        ReaderClass = SUPPORTED_READERS[args.reader]
        reader = ReaderClass()

        # --- 4. Initialize Writer ---
        log.info(f"Initializing writer: {args.writer}...")
        WriterClass = SUPPORTED_WRITERS[args.writer]
        writer = WriterClass()

        # --- 5. Run the "Read" Step ---
        log.info(f"Reading from source path: {args.input}")
        intermediate_data = reader.read(args.input)
        log.info("Successfully read and parsed source data.")

        # --- 6. Run the "Write" Step ---
        log.info(f"Writing to output path: {args.output}")
        writer.write(intermediate_data, args.output)
        
        log.info("=" * 70)
        log.info("Conversion finished successfully!")
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
