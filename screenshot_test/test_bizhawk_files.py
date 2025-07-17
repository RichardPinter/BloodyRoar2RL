import os
import time
import glob
from PIL import Image
import numpy as np

def test_read_bizhawk_frame(filepath):
    """Test reading a frame file created by BizHawk."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    start_time = time.perf_counter()
    
    try:
        # Read the image
        img = Image.open(filepath)
        
        # Convert to numpy array (like we'd do for RL)
        img_array = np.array(img)
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        print(f"Successfully read frame:")
        print(f"  File: {os.path.basename(filepath)}")
        print(f"  Size: {img.size}")
        print(f"  Array shape: {img_array.shape}")
        print(f"  Read time: {elapsed_ms:.3f}ms")
        
        return img_array, elapsed_ms
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def monitor_bizhawk_output():
    """Monitor for new frame files from BizHawk."""
    output_dir = "."  # Current directory
    pattern = "bizhawk_frame_*.png"
    
    print("Monitoring for BizHawk frame files...")
    print(f"Pattern: {pattern}")
    print("Run the BizHawk Lua script and this will detect new files\n")
    
    seen_files = set()
    read_times = []
    
    try:
        while True:
            # Look for new frame files
            files = glob.glob(pattern)
            
            for filepath in files:
                if filepath not in seen_files:
                    seen_files.add(filepath)
                    result = test_read_bizhawk_frame(filepath)
                    
                    if result:
                        img_array, read_time = result
                        read_times.append(read_time)
                        
                        # Show stats
                        if len(read_times) > 1:
                            avg_time = sum(read_times) / len(read_times)
                            print(f"  Average read time: {avg_time:.3f}ms")
                        print()
            
            time.sleep(0.1)  # Check every 100ms
            
    except KeyboardInterrupt:
        print("\nStopped monitoring")
        
        if read_times:
            print(f"\nFinal stats:")
            print(f"  Files read: {len(read_times)}")
            print(f"  Average read time: {sum(read_times)/len(read_times):.3f}ms")
            print(f"  Min read time: {min(read_times):.3f}ms")
            print(f"  Max read time: {max(read_times):.3f}ms")

def test_existing_files():
    """Test reading any existing BizHawk frame files."""
    pattern = "bizhawk_frame_*.png"
    files = glob.glob(pattern)
    
    if not files:
        print("No existing BizHawk frame files found")
        print(f"Looking for: {pattern}")
        print("\nRun the BizHawk Lua script first to generate test files")
        return
    
    print(f"Found {len(files)} existing frame files")
    
    read_times = []
    for filepath in sorted(files):
        result = test_read_bizhawk_frame(filepath)
        if result:
            _, read_time = result
            read_times.append(read_time)
    
    if read_times:
        print(f"\nSummary:")
        print(f"  Files read: {len(read_times)}")
        print(f"  Average read time: {sum(read_times)/len(read_times):.3f}ms")
        print(f"  Min read time: {min(read_times):.3f}ms")
        print(f"  Max read time: {max(read_times):.3f}ms")
        
        # Compare with screen capture
        print(f"\nComparison with screen capture:")
        print(f"  BizHawk file read: ~{sum(read_times)/len(read_times):.1f}ms")
        print(f"  Screen capture (MSS): ~16.6ms")
        
        if sum(read_times)/len(read_times) < 16.6:
            print(f"  ✓ BizHawk method is FASTER!")
        else:
            print(f"  ✗ BizHawk method is slower")

def main():
    print("BizHawk Frame File Reader Test")
    print("="*50)
    
    print("\n1. Testing existing files...")
    test_existing_files()
    
    print("\n" + "="*50)
    print("Instructions:")
    print("1. Open BizHawk")
    print("2. Load a game (like Atari 2600)")
    print("3. Open Tools -> Lua Console")
    print("4. Open the script: bizhawk_test.lua")
    print("5. Run it and check for generated frame files")
    print("\nThen run this script again to test file reading speeds")
    
    choice = input("\nMonitor for new files? (y/n): ")
    if choice.lower() == 'y':
        monitor_bizhawk_output()

if __name__ == "__main__":
    main()