import cv2
import numpy as np
from PIL import ImageGrab
import time

class SnakeGameAnalyzer:
    def __init__(self, x1=520, y1=225, x2=1405, y2=1005, grid_width=17, grid_height=15):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Calculate box dimensions
        self.box_width = (x2 - x1) / grid_width
        self.box_height = (y2 - y1) / grid_height
        
        # Color thresholds for different states (BGR format)
        # Adjusted for your specific Snake game colors
        self.color_ranges = {
            'apple': {
                'lower': np.array([0, 0, 150]),    # Red apple
                'upper': np.array([50, 50, 255])
            },
            'body': {
                'lower': np.array([80, 0, 0]),     # All tones of blue for body (including head)
                'upper': np.array([255, 100, 100])
            },
            'empty': {
                'lower': np.array([0, 100, 0]),    # Green empty spaces
                'upper': np.array([100, 255, 100])
            }
        }
    
    def capture_screen(self):
        """Capture the specified screen region"""
        screenshot = ImageGrab.grab(bbox=(self.x1, self.y1, self.x2, self.y2))
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    def get_box_center(self, row, col):
        """Get the center coordinates of a specific box"""
        center_x = int(col * self.box_width + self.box_width / 2)
        center_y = int(row * self.box_height + self.box_height / 2)
        return center_x, center_y
    
    def get_box_region(self, image, row, col):
        """Extract a small region around the center of a box for analysis"""
        center_x, center_y = self.get_box_center(row, col)
        
        # Sample a small region around the center (adjust size as needed)
        sample_size = 10
        x1 = max(0, center_x - sample_size)
        y1 = max(0, center_y - sample_size)
        x2 = min(image.shape[1], center_x + sample_size)
        y2 = min(image.shape[0], center_y + sample_size)
        
        return image[y1:y2, x1:x2]
    
    def analyze_box_color(self, box_region):
        """Analyze the dominant color in a box region"""
        if box_region.size == 0:
            return 'empty'
        
        # Calculate average color
        avg_color = np.mean(box_region, axis=(0, 1))
        
        # Check for body (all tones of blue, including head)
        if avg_color[0] > 80 and avg_color[1] < 100 and avg_color[2] < 100:
            return 'body'
        
        # Check for empty (green)
        if avg_color[1] > 100 and avg_color[0] < 100 and avg_color[2] < 100:
            return 'empty'
        
        # Check for apple (red)
        if avg_color[2] > 150 and avg_color[0] < 50 and avg_color[1] < 50:
            return 'apple'
        
        # Fallback to original color range checking
        for state, color_range in self.color_ranges.items():
            lower = color_range['lower']
            upper = color_range['upper']
            
            if np.all(avg_color >= lower) and np.all(avg_color <= upper):
                return state
        
        # If no exact match, find closest match based on dominant color
        if avg_color[0] > max(avg_color[1], avg_color[2]):  # Blue dominant
            return 'body'
        elif avg_color[1] > max(avg_color[0], avg_color[2]):  # Green dominant
            return 'empty'
        elif avg_color[2] > max(avg_color[0], avg_color[1]):  # Red dominant
            return 'apple'
        else:
            return 'empty'
    
    def analyze_grid(self):
        """Analyze the entire grid and return the state of each box"""
        image = self.capture_screen()
        grid_state = []
        
        for row in range(self.grid_height):
            row_state = []
            for col in range(self.grid_width):
                box_region = self.get_box_region(image, row, col)
                state = self.analyze_box_color(box_region)
                row_state.append(state)
            grid_state.append(row_state)
        
        return grid_state
    
    def print_grid(self, grid_state):
        """Print the grid state in a readable format"""
        symbols = {
            'empty': '.',
            'apple': 'A',
            'body': 'B'
        }
        
        print("Snake Game Grid State:")
        print("-" * (self.grid_width * 2 + 1))
        
        for row in grid_state:
            print("|", end="")
            for cell in row:
                print(symbols.get(cell, '?'), end=" ")
            print("|")
        
        print("-" * (self.grid_width * 2 + 1))
    
    def print_detailed_grid(self, grid_state):
        """Print detailed grid with coordinates and states"""
        print("\nDetailed Grid Analysis:")
        print(f"Grid size: {self.grid_width}x{self.grid_height}")
        print(f"Box dimensions: {self.box_width:.1f}x{self.box_height:.1f}")
        print()
        
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                state = grid_state[row][col]
                print(f"Box ({row:2d},{col:2d}): {state:5s}", end="  ")
                if (col + 1) % 6 == 0:  # New line every 6 boxes for readability
                    print()
            print()
    
    def calibrate_colors(self):
        """Helper function to calibrate colors by sampling the current screen"""
        print("Color Calibration Mode")
        print("Click on different elements in the game to sample their colors...")
        print("This will help adjust the color ranges for better detection.")
        
        image = self.capture_screen()
        
        # Sample colors from different regions
        print("\nSample some colors from the current screen:")
        for row in range(0, self.grid_height, 3):
            for col in range(0, self.grid_width, 3):
                box_region = self.get_box_region(image, row, col)
                if box_region.size > 0:
                    avg_color = np.mean(box_region, axis=(0, 1))
                    print(f"Box ({row:2d},{col:2d}): BGR({avg_color[0]:3.0f}, {avg_color[1]:3.0f}, {avg_color[2]:3.0f})")

def main():
    # Initialize the analyzer
    analyzer = SnakeGameAnalyzer()
    
    print("Snake Game State Analyzer")
    print("=" * 50)
    print(f"Monitoring region: ({analyzer.x1}, {analyzer.y1}) to ({analyzer.x2}, {analyzer.y2})")
    print(f"Grid size: {analyzer.grid_width}x{analyzer.grid_height}")
    print(f"Box size: {analyzer.box_width:.1f}x{analyzer.box_height:.1f}")
    print()
    
    # Uncomment the line below to calibrate colors first
    # analyzer.calibrate_colors()
    
    try:
        while True:
            # Analyze the current grid state
            grid_state = analyzer.analyze_grid()
            
            # Clear screen (optional, comment out if not needed)
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Print the grid
            analyzer.print_grid(grid_state)
            
            # Print detailed analysis (comment out if not needed)
            # analyzer.print_detailed_grid(grid_state)
            
            # Wait before next analysis
            time.sleep(0.5)  # Adjust refresh rate as needed
            
    except KeyboardInterrupt:
        print("\nStopping analyzer...")

if __name__ == "__main__":
    main()