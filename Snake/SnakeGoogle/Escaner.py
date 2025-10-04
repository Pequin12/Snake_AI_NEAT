import numpy as np
from PIL import Image, ImageGrab
import cv2

def scan_screen_region(x, y, width, height, scale_factor=3):
    """
    Scan a region of the screen and process it to grayscale with lower resolution
    
    Args:
        x, y: Top-left coordinates of the region to capture
        width, height: Dimensions of the region to capture
        scale_factor: How much to reduce the resolution (e.g., 3 = 3 times smaller)
    
    Returns:
        numpy array with values from 0 (white) to 10 (black)
    """
    
    # Capture the screen region
    bbox = (x, y, x + width, y + height)
    screenshot = ImageGrab.grab(bbox)
    
    # Convert to grayscale
    grayscale = screenshot.convert('L')
    
    # Calculate target dimensions based on scale factor
    target_width = max(1, width // scale_factor)
    target_height = max(1, height // scale_factor)
    
    # Resize to lower resolution
    low_res = grayscale.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    pixel_array = np.array(low_res)
    
    # Map from 0-255 grayscale to 0-10 scale (inverted: 0=white, 10=black)
    # Formula: value = 10 - (pixel_value / 255 * 10)
    scaled_array = 10 - (pixel_array / 255 * 10)
    scaled_array = np.round(scaled_array).astype(int)
    
    return scaled_array



def save_visualization(pixel_array, filename="screen_scan.png"):
    """
    Save a visualization of the processed image
    """
    # Convert back to 0-255 scale for saving
    visual_array = (10 - pixel_array) * 25.5
    visual_array = visual_array.astype(np.uint8)
    
    # Create and save image
    img = Image.fromarray(visual_array, mode='L')
    img.save(filename)
    print(f"Visualization saved as {filename}")

# Example usage
if __name__ == "__main__":
    # Example: Scan a 200x200 region starting at coordinates (100, 100)
    # and convert to 20x20 low resolution
    
    print("Screen Region Scanner")
    print("=" * 50)
    
    # Get user input for region coordinates
    try:
        x = int(input("Enter X coordinate (top-left): "))
        y = int(input("Enter Y coordinate (top-left): "))
        width = int(input("Enter width: "))
        height = int(input("Enter height: "))
        
        # Optional: custom scale factor
        scale_factor = input("Enter scale factor (default 3): ")
        scale_factor = int(scale_factor) if scale_factor else 3
        
        print(f"\nScanning region: ({x}, {y}) with size {width}x{height}")
        print(f"Scale factor: {scale_factor} (output will be {width//scale_factor}x{height//scale_factor})")
        
        # Process the screen region
        result = scan_screen_region(x, y, width, height, scale_factor)
        
        print(f"\nProcessed array shape: {result.shape}")
        
        # Save visualization
        save_visualization(result)
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the coordinates are within your screen bounds.")

# Alternative function for automated scanning with predefined settings
def quick_scan(x=100, y=100, width=200, height=200, scale_factor=3):
    """
    Quick scan function with default parameters
    """
    result = scan_screen_region(x, y, width, height, scale_factor)
    output_w = width // scale_factor
    output_h = height // scale_factor
    print(f"Quick scan result ({output_w}x{output_h}):")
    save_visualization(result)
    return result