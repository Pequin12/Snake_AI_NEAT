import cv2
import numpy as np
from PIL import ImageGrab
import time
import neat
import os
import pickle
import pyautogui
import threading
import queue

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
        
        # Sample a small region around the center
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
    
    def get_flattened_state(self):
        """Get the grid state as a flattened array of numerical values"""
        grid_state = self.analyze_grid()
        
        # Convert states to numbers for neural network input
        state_to_num = {
            'empty': 0,
            'apple': 1,
            'body': 2
        }
        
        # Flatten the grid and convert to numbers
        flattened = []
        for row in grid_state:
            for cell in row:
                flattened.append(state_to_num.get(cell, 0))
        
        return np.array(flattened, dtype=np.float32)
    
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

class SnakeAI:
    def __init__(self, config_path='config-feedforward.txt'):
        self.config_path = config_path
        self.capture_interval = 0.05  # 50ms
        self.running = False
        self.current_generation = 0
        self.best_fitness = 0
        self.current_individual = 0
        self.total_individuals = 0
        self.generation_fitnesses = []
        
        # Initialize game analyzer
        self.game_analyzer = SnakeGameAnalyzer()
        
        # Game state tracking
        self.game_start_time = 0
        self.last_game_state = None
        self.death_detected = False
        self.survival_time = 0
        self.last_score = 0
        
        # Screen capture queue
        self.screen_queue = queue.Queue(maxsize=10)
        self.capture_thread = None
        
        # Create NEAT configuration
        self.create_neat_config()
        
    def create_neat_config(self):
        """Create NEAT configuration file"""
        # Calculate total inputs (17 x 15 = 255 boxes)
        total_inputs = self.game_analyzer.grid_width * self.game_analyzer.grid_height
        
        config_content = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000
pop_size              = 100
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.05
activation_options      = sigmoid relu tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.05
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = {total_inputs}
num_outputs             = 4

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def capture_game_state_continuous(self):
        """Continuously capture game state in a separate thread"""
        while self.running:
            try:
                # Get current game state
                current_state = self.game_analyzer.get_flattened_state()
                
                # Add to queue (non-blocking)
                if not self.screen_queue.full():
                    self.screen_queue.put(current_state)
                
                time.sleep(self.capture_interval)
            except Exception as e:
                print(f"Game state capture error: {e}")
                time.sleep(0.1)
    
    def detect_death(self, current_state, previous_state):
        """Detect if the snake died by checking if 30% of the screen changed rapidly"""
        if previous_state is None:
            return False
        
        # Check for sudden major changes in game state (30% threshold)
        state_diff = np.sum(np.abs(current_state - previous_state))
        if state_diff > len(current_state) * 0.3:  # 30% of state changed
            return True
        
        return False
    
    def calculate_score(self, current_state):
        """Calculate current score based on body segments"""
        body_count = np.sum(current_state == 2)
        return max(0, body_count - 3)  # Assuming initial snake has 3 segments
    
    def send_input(self, action):
        """Send keyboard input based on AI decision"""
        # Actions: 0=up, 1=down, 2=left, 3=right
        key_map = {
            0: 'up',
            1: 'down', 
            2: 'left',
            3: 'right'
        }
        
        if action in key_map:
            pyautogui.press(key_map[action])
    
    def restart_game(self):
        """Restart the game by pressing space, then right arrow to start movement"""
        # Wait 0.6 seconds before pressing space to ensure we start the game correctly 
        time.sleep(0.6)
        
        # Press space to restart
        pyautogui.press('space')
        
        # Immediately press right arrow to start the snake moving
        pyautogui.press('right')
        
        # Reset tracking variables
        self.game_start_time = time.time()
        self.death_detected = False
        self.last_score = 0
        
        # Clear the screen queue to avoid old state data
        while not self.screen_queue.empty():
            try:
                self.screen_queue.get_nowait()
            except queue.Empty:
                break
    
    def print_agent_header(self, genome_id):
        """Print detailed header for current agent"""
        print(f"\n{'='*80}")
        print(f"🤖 AGENT EVALUATION")
        print(f"{'='*80}")
        print(f"Generation: {self.current_generation}")
        print(f"Individual: {self.current_individual}/{self.total_individuals}")
        print(f"Genome ID: {genome_id}")
        print(f"Current Best Fitness (All Generations): {self.best_fitness:.2f}")
        if self.generation_fitnesses:
            print(f"Best Fitness This Generation: {max(self.generation_fitnesses):.2f}")
            print(f"Average Fitness This Generation: {np.mean(self.generation_fitnesses):.2f}")
        print(f"{'='*80}")
    
    def print_agent_death_summary(self, genome_id, fitness, frames_alive, max_score):
        """Print summary when agent dies"""
        print(f"\n{'='*80}")
        print(f"💀 AGENT DEATH SUMMARY")
        print(f"{'='*80}")
        print(f"Generation: {self.current_generation}")
        print(f"Individual: {self.current_individual}/{self.total_individuals}")
        print(f"Genome ID: {genome_id}")
        print(f"Final Fitness: {fitness:.2f}")
        print(f"Frames Survived: {frames_alive}")
        print(f"Max Score Achieved: {max_score}")
        print(f"Current Best Fitness (All Generations): {self.best_fitness:.2f}")
        
        # Show performance relative to generation
        if self.generation_fitnesses:
            gen_avg = np.mean(self.generation_fitnesses)
            gen_best = max(self.generation_fitnesses)
            performance_vs_avg = ((fitness - gen_avg) / gen_avg * 100) if gen_avg > 0 else 0
            performance_vs_best = ((fitness - gen_best) / gen_best * 100) if gen_best > 0 else 0
            
            print(f"Performance vs Generation Average: {performance_vs_avg:+.1f}%")
            print(f"Performance vs Generation Best: {performance_vs_best:+.1f}%")
        
        # Show if this is a new record
        if fitness > self.best_fitness:
            print(f"🎉 NEW OVERALL RECORD! Previous best: {self.best_fitness:.2f}")
        
        print(f"{'='*80}")
    
    def evaluate_genome(self, genome, config, genome_id):
        """Evaluate a single genome"""
        # Print agent header
        self.print_agent_header(genome_id)
        
        # Create neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Reset game
        self.restart_game()
        
        fitness = 0
        frames_alive = 0
        max_frames = 2000  # Reduced for faster training
        score = 0
        max_score = 0
        
        previous_state = None
        frames_since_restart = 0
        
        # Progress tracking
        print(f"🏁 Starting evaluation... (max frames: {max_frames})")
        last_progress_report = 0
        
        while frames_alive < max_frames and self.running:
            try:
                # Get current game state
                if not self.screen_queue.empty():
                    current_state = self.screen_queue.get_nowait()
                    
                    # Skip first few frames after restart to let game stabilize
                    if frames_since_restart < 10:
                        frames_since_restart += 1
                        previous_state = current_state.copy()
                        continue
                    
                    # Check for death
                    if self.detect_death(current_state, previous_state):
                        print(f"💀 Death detected at frame {frames_alive}")
                        # Print death summary
                        self.print_agent_death_summary(genome_id, fitness, frames_alive, max_score)
                        break
                    
                    # Calculate current score
                    score = self.calculate_score(current_state)
                    if score > max_score:
                        max_score = score
                        fitness += 100  # Bonus for eating apples
                        print(f"🍎 Apple eaten! Score: {score}, Current Fitness: {fitness:.2f}")
                    
                    # Get AI decision
                    output = net.activate(current_state)
                    action = np.argmax(output)
                    
                    # Send input
                    self.send_input(action)
                    
                    # Update fitness (reward for staying alive)
                    fitness += 1
                    
                    # Bonus for being near apples
                    apple_positions = np.where(current_state == 1)[0]
                    if len(apple_positions) > 0:
                        # Simple proximity bonus (this is a basic implementation)
                        fitness += 0.1
                    
                    frames_alive += 1
                    frames_since_restart += 1
                    previous_state = current_state.copy()
                    
                    # Progress report every 500 frames
                    if frames_alive - last_progress_report >= 500:
                        print(f"📊 Progress: {frames_alive}/{max_frames} frames, Score: {score}, Fitness: {fitness:.2f}")
                        last_progress_report = frames_alive
                
                # Small delay to control game speed
                time.sleep(0.02)  # Faster evaluation
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in evaluation: {e}")
                break
        
        # Final fitness calculation
        fitness += frames_alive * 0.1  # Survival bonus
        fitness += max_score * 1000    # Score bonus (eating apples)
        
        # Penalty for dying early
        if frames_alive < 100:
            fitness *= 0.5
        
        genome.fitness = fitness
        
        # Print final death summary if not already printed
        if frames_alive >= max_frames:
            print(f"⏰ Time limit reached!")
            self.print_agent_death_summary(genome_id, fitness, frames_alive, max_score)
        
        return fitness
    
    def run_generation(self, genomes, config):
        """Run a generation of genomes"""
        print(f"\n{'='*80}")
        print(f"🧬 GENERATION {self.current_generation} STARTED")
        print(f"{'='*80}")
        print(f"Population Size: {len(genomes)}")
        print(f"Overall Best Fitness: {self.best_fitness:.2f}")
        print(f"{'='*80}")
        
        self.generation_fitnesses = []
        generation_best_fitness = 0
        self.total_individuals = len(genomes)
        
        for i, (genome_id, genome) in enumerate(genomes, 1):
            self.current_individual = i
            
            fitness = self.evaluate_genome(genome, config, genome_id)
            self.generation_fitnesses.append(fitness)
            
            if fitness > generation_best_fitness:
                generation_best_fitness = fitness
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                print(f"\n🎉 NEW OVERALL BEST FITNESS: {fitness:.2f}")
                
                # Save best genome
                with open(f'best_genome_gen_{self.current_generation}.pkl', 'wb') as f:
                    pickle.dump(genome, f)
        
        # Generation summary
        print(f"\n{'='*80}")
        print(f"📊 GENERATION {self.current_generation} COMPLETED")
        print(f"{'='*80}")
        print(f"Best fitness this generation: {generation_best_fitness:.2f}")
        print(f"Average fitness this generation: {np.mean(self.generation_fitnesses):.2f}")
        print(f"Worst fitness this generation: {min(self.generation_fitnesses):.2f}")
        print(f"Fitness std deviation: {np.std(self.generation_fitnesses):.2f}")
        print(f"Overall best fitness: {self.best_fitness:.2f}")
        
        # Show improvement
        if self.current_generation > 0:
            improvement = generation_best_fitness - (self.best_fitness if generation_best_fitness <= self.best_fitness else 0)
            print(f"Generation improvement: {improvement:.2f}")
        
        print(f"{'='*80}")
        
        self.current_generation += 1
    
    def run(self):
        """Main execution function"""
        print("🐍 NEAT Snake AI - Enhanced Reporting System")
        print("=" * 60)
        print("Make sure your Snake game is visible and positioned correctly.")
        print("The AI will start learning in 5 seconds...")
        
        # Wait for user to position game window
        for i in range(5, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
        
        self.running = True
        
        # Start game state capture thread
        self.capture_thread = threading.Thread(target=self.capture_game_state_continuous)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Load NEAT configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.config_path)
        
        # Create population
        population = neat.Population(config)
        
        # Add reporters
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Custom reporter for species information
        class CustomReporter(neat.reporting.BaseReporter):
            def __init__(self, snake_ai_instance):
                self.snake_ai = snake_ai_instance
                
            def post_evaluate(self, config, population, species, best_genome):
                self.snake_ai.print_species_report(species, best_genome)
        
        population.add_reporter(CustomReporter(self))
        
        # Run evolution
        try:
            winner = population.run(self.run_generation, 50)  # 50 generations
            
            # Save winner
            with open('winner_genome.pkl', 'wb') as f:
                pickle.dump(winner, f)
            
            print(f"\n🏆 Evolution completed! Best fitness: {self.best_fitness:.2f}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Evolution interrupted by user")
        finally:
            self.running = False
            if self.capture_thread:
                self.capture_thread.join(timeout=1)
    
    def play_with_trained_ai(self, genome_file='winner_genome.pkl'):
        """Play the game with a trained AI"""
        if not os.path.exists(genome_file):
            print(f"Genome file {genome_file} not found!")
            return
        
        # Load trained genome
        with open(genome_file, 'rb') as f:
            genome = pickle.load(f)
        
        # Load configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.config_path)
        
        # Create network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        print("🎮 Playing with trained AI...")
        print("Press Ctrl+C to stop")
        
        self.running = True
        
        # Start game state capture
        self.capture_thread = threading.Thread(target=self.capture_game_state_continuous)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Reset game
        self.restart_game()
        
        previous_state = None
        frames_since_restart = 0
        
        try:
            while self.running:
                if not self.screen_queue.empty():
                    current_state = self.screen_queue.get_nowait()
                    
                    # Skip first few frames after restart
                    if frames_since_restart < 10:
                        frames_since_restart += 1
                        previous_state = current_state.copy()
                        continue
                    
                    # Check for death and auto-restart
                    if self.detect_death(current_state, previous_state):
                        print("💀 Death detected - auto-restarting...")
                        self.restart_game()
                        frames_since_restart = 0
                        previous_state = None
                        continue
                    
                    # Get AI decision
                    output = net.activate(current_state)
                    action = np.argmax(output)
                    
                    # Send input
                    self.send_input(action)
                    
                    # Print current grid state occasionally
                    if np.random.random() < 0.1:  # 10% chance to print
                        grid_state = self.game_analyzer.analyze_grid()
                        self.game_analyzer.print_grid(grid_state)
                        print(f"Score: {self.calculate_score(current_state)}")
                    
                    previous_state = current_state.copy()
                    frames_since_restart += 1
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n⚠️ Stopped by user")
        finally:
            self.running = False

def main():
    """Main function"""
    print("🐍 NEAT Snake AI - Enhanced Reporting System")
    print("=" * 60)
    print("1. Train new AI")
    print("2. Play with trained AI")
    print("3. Test game analyzer")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        snake_ai = SnakeAI()
        snake_ai.run()
    elif choice == '2':
        snake_ai = SnakeAI()
        snake_ai.play_with_trained_ai()
    elif choice == '3':
        # Test the game analyzer
        analyzer = SnakeGameAnalyzer()
        print("Testing game analyzer...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                grid_state = analyzer.analyze_grid()
                os.system('cls' if os.name == 'nt' else 'clear')
                analyzer.print_grid(grid_state)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    # Install required packages
    required_packages = [
        'opencv-python', 'numpy', 'pyautogui', 
        'neat-python', 'pillow'
    ]
    
    print("Required packages:", ", ".join(required_packages))
    print("Install with: pip install " + " ".join(required_packages))
    print()
    
    main()