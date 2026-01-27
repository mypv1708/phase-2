#!/usr/bin/env python3
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.logging_config import setup_logging
from nodes.perception.mic_driver_node import MicDriverNode


def main():
    setup_logging()
    
    print("=" * 60)
    print("Mic Driver - Wake Word Detection & Audio Enhancement")
    print("=" * 60)
    print()
    
    try:
        node = MicDriverNode()
        node.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()