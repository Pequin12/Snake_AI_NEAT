# -*- coding: utf-8 -*-
import pygame
import neat
import os
import random
import pickle
import numpy as np
from enum import Enum
import sys
import time
import pandas as pd
from datetime import datetime
import copy
import json

# --- CONSTANTES Y CONFIGURACIÓN INICIAL ---
# Dimensiones de la cuadrícula y la ventana
GRID_SIZE = 25
GRID_WIDTH = 20
GRID_HEIGHT = 15
WINDOW_WIDTH = GRID_WIDTH * GRID_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * GRID_SIZE

# Fotogramas por segundo (para la visualización)
FPS = 60

# Colores
BLACK = (0, 0, 0)
GREEN = (45, 200, 50)  # Cabeza
BLUE = (30, 144, 255)  # Cuerpo
RED = (255, 50, 50)    # Comida
WHITE = (255, 255, 255)

# --- CLASE PARA RECOPILAR ESTADÍSTICAS ---
class NEATStatsCollector:
    """Recopila estadísticas detalladas del entrenamiento NEAT."""
    
    def __init__(self):
        self.training_start_time = None
        self.training_end_time = None
        self.generation_data = []
        self.champion_history = []
        self.population_config = {}
        self.mutation_config = {}
        self.species_history = []
        self.best_genome_global = None
        self.best_fitness_global = float('-inf')
        self.generation_times = []
        
    def start_training(self, config):
        """Inicia el registro del entrenamiento."""
        self.training_start_time = datetime.now()
        self.population_config = {
            'pop_size': config.pop_size,
            'num_inputs': config.genome_config.num_inputs,
            'num_outputs': config.genome_config.num_outputs,
            'num_hidden': config.genome_config.num_hidden,
            'initial_connection': config.genome_config.initial_connection,
            'feed_forward': config.genome_config.feed_forward
        }
        
        self.mutation_config = {
            'conn_add_prob': config.genome_config.conn_add_prob,
            'conn_delete_prob': config.genome_config.conn_delete_prob,
            'node_add_prob': config.genome_config.node_add_prob,
            'node_delete_prob': config.genome_config.node_delete_prob,
            'weight_mutate_rate': config.genome_config.weight_mutate_rate,
            'weight_init_mean': config.genome_config.weight_init_mean,
            'weight_init_stdev': config.genome_config.weight_init_stdev,
            'weight_max_value': config.genome_config.weight_max_value,
            'weight_min_value': config.genome_config.weight_min_value,
            'bias_mutate_rate': config.genome_config.bias_mutate_rate,
            'enabled_mutate_rate': config.genome_config.enabled_mutate_rate
        }
    
    def end_training(self):
        """Finaliza el registro del entrenamiento."""
        self.training_end_time = datetime.now()
    
    def record_generation(self, generation, population, species_set, best_genome):
        """Registra datos de una generación."""
        gen_start_time = time.time()
        
        # Calcular estadísticas de fitness
        fitnesses = [genome.fitness for genome in population.values()]
        fitness_avg = np.mean(fitnesses)
        fitness_max = np.max(fitnesses)
        fitness_min = np.min(fitnesses)
        
        # Actualizar mejor genoma global
        if fitness_max > self.best_fitness_global:
            self.best_fitness_global = fitness_max
            self.best_genome_global = copy.deepcopy(best_genome)
        
        # Calcular estadísticas de topología
        node_counts = [len(genome.nodes) for genome in population.values()]
        connection_counts = [len(genome.connections) for genome in population.values()]
        
        # Información de especies
        species_info = []
        for species_id, species in species_set.species.items():
            species_info.append({
                'id': species_id,
                'size': len(species.members),
                'fitness': species.fitness,
                'stagnation': species.fitness_history[-1] if species.fitness_history else 0
            })
        
        # Análisis del campeón
        champion_info = self._analyze_genome(best_genome)
        
        # Datos de la generación
        gen_data = {
            'generation': generation,
            'fitness_avg': fitness_avg,
            'fitness_max': fitness_max,
            'fitness_min': fitness_min,
            'fitness_std': np.std(fitnesses),
            'num_species': len(species_set.species),
            'avg_nodes': np.mean(node_counts),
            'avg_connections': np.mean(connection_counts),
            'max_nodes': np.max(node_counts),
            'max_connections': np.max(connection_counts),
            'min_nodes': np.min(node_counts),
            'min_connections': np.min(connection_counts),
            'champion_fitness': best_genome.fitness,
            'champion_nodes': len(best_genome.nodes),
            'champion_connections': len(best_genome.connections),
            'champion_active_connections': sum(1 for c in best_genome.connections.values() if c.enabled)
        }
        
        self.generation_data.append(gen_data)
        self.champion_history.append(champion_info)
        self.species_history.append(species_info)
        
        gen_end_time = time.time()
        self.generation_times.append(gen_end_time - gen_start_time)
    
    def _analyze_genome(self, genome):
        """Analiza un genoma individual."""
        active_connections = sum(1 for c in genome.connections.values() if c.enabled)
        inactive_connections = len(genome.connections) - active_connections
        
        return {
            'genome_id': genome.key,
            'fitness': genome.fitness,
            'nodes': len(genome.nodes),
            'total_connections': len(genome.connections),
            'active_connections': active_connections,
            'inactive_connections': inactive_connections,
            'node_genes': list(genome.nodes.keys()),
            'connection_genes': [(c.key, c.enabled, c.weight) for c in genome.connections.values()]
        }
    
    def generate_excel_report(self, filename="snake_ai_training_report.xlsx"):
        """Genera un reporte completo en Excel."""
        print("📊 Generando reporte de entrenamiento...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Hoja 1: Resumen del entrenamiento
            self._create_training_summary_sheet(writer)
            
            # Hoja 2: Configuración NEAT
            self._create_config_sheet(writer)
            
            # Hoja 3: Datos por generación
            self._create_generation_data_sheet(writer)
            
            # Hoja 4: Historial de campeones
            self._create_champion_history_sheet(writer)
            
            # Hoja 5: Evolución de especies
            self._create_species_evolution_sheet(writer)
            
            # Hoja 6: Análisis del mejor genoma
            self._create_best_genome_analysis_sheet(writer)
            
            # Hoja 7: Estadísticas de mutaciones
            self._create_mutation_stats_sheet(writer)
        
        print(f"✅ Reporte generado: {filename}")
    
    def _create_training_summary_sheet(self, writer):
        """Crea la hoja de resumen del entrenamiento."""
        if not self.generation_data:
            return
            
        training_time = (self.training_end_time - self.training_start_time).total_seconds()
        
        summary_data = {
            'Métrica': [
                'Fecha de inicio', 'Fecha de fin', 'Duración total (segundos)',
                'Generaciones completadas', 'Tiempo promedio por generación',
                'Mejor fitness global', 'Fitness promedio final',
                'Población final', 'Especies finales',
                'Nodos promedio finales', 'Conexiones promedio finales'
            ],
            'Valor': [
                self.training_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                self.training_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                round(training_time, 2),
                len(self.generation_data),
                round(np.mean(self.generation_times), 4),
                self.best_fitness_global,
                self.generation_data[-1]['fitness_avg'],
                self.population_config['pop_size'],
                self.generation_data[-1]['num_species'],
                round(self.generation_data[-1]['avg_nodes'], 2),
                round(self.generation_data[-1]['avg_connections'], 2)
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Resumen', index=False)
    
    def _create_config_sheet(self, writer):
        """Crea la hoja de configuración NEAT."""
        # Configuración de población
        pop_data = []
        for key, value in self.population_config.items():
            pop_data.append(['Población', key, value])
        
        # Configuración de mutaciones
        for key, value in self.mutation_config.items():
            pop_data.append(['Mutaciones', key, value])
        
        df_config = pd.DataFrame(pop_data, columns=['Categoría', 'Parámetro', 'Valor'])
        df_config.to_excel(writer, sheet_name='Configuración', index=False)
    
    def _create_generation_data_sheet(self, writer):
        """Crea la hoja de datos por generación."""
        if not self.generation_data:
            return
            
        df_generations = pd.DataFrame(self.generation_data)
        df_generations.to_excel(writer, sheet_name='Datos por Generación', index=False)
    
    def _create_champion_history_sheet(self, writer):
        """Crea la hoja de historial de campeones."""
        if not self.champion_history:
            return
            
        champion_data = []
        for i, champion in enumerate(self.champion_history):
            champion_data.append({
                'Generación': i,
                'ID_Genoma': champion['genome_id'],
                'Fitness': champion['fitness'],
                'Nodos': champion['nodes'],
                'Conexiones_Totales': champion['total_connections'],
                'Conexiones_Activas': champion['active_connections'],
                'Conexiones_Inactivas': champion['inactive_connections']
            })
        
        df_champions = pd.DataFrame(champion_data)
        df_champions.to_excel(writer, sheet_name='Historial Campeones', index=False)
    
    def _create_species_evolution_sheet(self, writer):
        """Crea la hoja de evolución de especies."""
        if not self.species_history:
            return
            
        species_data = []
        for gen, species_list in enumerate(self.species_history):
            for species in species_list:
                species_data.append({
                    'Generación': gen,
                    'ID_Especie': species['id'],
                    'Tamaño': species['size'],
                    'Fitness': species['fitness'],
                    'Estancamiento': species['stagnation']
                })
        
        df_species = pd.DataFrame(species_data)
        df_species.to_excel(writer, sheet_name='Evolución Especies', index=False)
    
    def _create_best_genome_analysis_sheet(self, writer):
        """Crea la hoja de análisis del mejor genoma."""
        if not self.best_genome_global:
            return
            
        best_analysis = self._analyze_genome(self.best_genome_global)
        
        # Información básica
        basic_info = {
            'Propiedad': ['ID del Genoma', 'Fitness', 'Nodos', 'Conexiones Totales', 
                         'Conexiones Activas', 'Conexiones Inactivas'],
            'Valor': [
                best_analysis['genome_id'],
                best_analysis['fitness'],
                best_analysis['nodes'],
                best_analysis['total_connections'],
                best_analysis['active_connections'],
                best_analysis['inactive_connections']
            ]
        }
        
        df_best_basic = pd.DataFrame(basic_info)
        df_best_basic.to_excel(writer, sheet_name='Mejor Genoma', index=False, startrow=0)
        
        # Detalles de conexiones
        connection_details = []
        for conn_key, enabled, weight in best_analysis['connection_genes']:
            connection_details.append({
                'Origen': conn_key[0],
                'Destino': conn_key[1],
                'Activa': enabled,
                'Peso': weight
            })
        
        if connection_details:
            df_connections = pd.DataFrame(connection_details)
            df_connections.to_excel(writer, sheet_name='Mejor Genoma', index=False, startrow=len(basic_info) + 3)
    
    def _create_mutation_stats_sheet(self, writer):
        """Crea la hoja de estadísticas de mutaciones."""
        mutation_stats = []
        for param, value in self.mutation_config.items():
            mutation_stats.append({
                'Parámetro': param,
                'Valor': value,
                'Descripción': self._get_mutation_description(param)
            })
        
        df_mutations = pd.DataFrame(mutation_stats)
        df_mutations.to_excel(writer, sheet_name='Estadísticas Mutaciones', index=False)
    
    def _get_mutation_description(self, param):
        """Devuelve descripción de parámetros de mutación."""
        descriptions = {
            'conn_add_prob': 'Probabilidad de agregar conexión',
            'conn_delete_prob': 'Probabilidad de eliminar conexión',
            'node_add_prob': 'Probabilidad de agregar nodo',
            'node_delete_prob': 'Probabilidad de eliminar nodo',
            'weight_mutate_rate': 'Tasa de mutación de pesos',
            'weight_init_mean': 'Media inicial de pesos',
            'weight_init_stdev': 'Desviación estándar inicial de pesos',
            'weight_max_value': 'Valor máximo de peso',
            'weight_min_value': 'Valor mínimo de peso',
            'bias_mutate_rate': 'Tasa de mutación de bias',
            'enabled_mutate_rate': 'Tasa de mutación de habilitación'
        }
        return descriptions.get(param, 'Sin descripción')

# --- CLASES DEL JUEGO (SIN CAMBIOS) ---

class Direction(Enum):
    """Enumeración para las direcciones posibles."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Snake:
    """Gestiona el estado, movimiento y lógica de la serpiente."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia la serpiente a su estado inicial."""
        start_x = GRID_WIDTH // 2
        start_y = GRID_HEIGHT // 2
        
        # Crea un cuerpo inicial de 3 segmentos en lugar de 1
        self.body = [
            (start_x, start_y),      # Cabeza
            (start_x - 1, start_y),  # Segmento 2
            (start_x - 2, start_y)   # Segmento 3
        ]
        
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.grow_next = False
        self.alive = True
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.max_steps_without_food = GRID_WIDTH * GRID_HEIGHT

    def set_direction(self, direction: Direction):
        """Registra la próxima dirección deseada para el siguiente movimiento."""
        self.next_direction = direction

    def move(self):
        """
        Realiza un movimiento de una casilla.
        Primero, valida y aplica la dirección registrada (`next_direction`).
        Luego, mueve la cabeza y gestiona las colisiones.
        """
        if not self.alive:
            return

        # 1. Validar y aplicar la dirección registrada para este movimiento
        opposites = {
            Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
        }
        is_reversal = (len(self.body) > 1 and
                       opposites.get(self.direction) == self.next_direction)

        if not is_reversal:
            self.direction = self.next_direction

        # 2. Calcular la nueva posición de la cabeza
        head_x, head_y = self.body[0]
        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        else: # Direction.RIGHT
            new_head = (head_x + 1, head_y)

        # 3. Comprobar colisiones
        # Colisión con las paredes
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            self.alive = False
            return
        # Colisión consigo misma
        if new_head in self.body:
            self.alive = False
            return

        # 4. Mover la serpiente
        self.body.insert(0, new_head)
        if self.grow_next:
            self.grow_next = False
        else:
            self.body.pop()

        # 5. Actualizar contadores
        self.steps += 1
        self.steps_since_food += 1
        if self.steps_since_food > self.max_steps_without_food:
            self.alive = False

    def eat_food(self):
        """Marca a la serpiente para que crezca en el siguiente movimiento."""
        self.grow_next = True
        self.score += 1
        self.steps_since_food = 0

    def get_head(self):
        """Devuelve la posición de la cabeza de la serpiente."""
        return self.body[0]

class Game:
    """
    Simulador principal del juego. Gestiona el estado del juego, la serpiente,
    la comida y el bucle de actualización.
    """
    def __init__(self, display=False):
        self.display = display
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()
            self.move_interval = 150  # 0.15 segundos por movimiento
            self.last_move_time = 0

        self.snake = Snake()
        self.food = self._generate_food()
        self.game_over = False

    def _generate_food(self):
        """Genera una nueva posición para la comida fuera del cuerpo de la serpiente."""
        while True:
            food_pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food_pos not in self.snake.body:
                return food_pos

    def update(self, action: int):
        """
        Avanza un paso en la simulación del juego.
        `action` es un entero (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT).
        """
        if self.game_over:
            return

        # 1. Registrar la acción de la IA como la próxima dirección
        self.snake.set_direction(Direction(action))

        # 2. Mover la serpiente (con control de tiempo si hay pantalla)
        if self.display:
            now = pygame.time.get_ticks()
            if now - self.last_move_time > self.move_interval:
                self.last_move_time = now
                self.snake.move()
        else:
            # Sin pantalla (entrenamiento), mover en cada llamada
            self.snake.move()

        # 3. Comprobar si la serpiente ha muerto
        if not self.snake.alive:
            self.game_over = True
            return

        # 4. Comprobar si la serpiente ha comido
        if self.snake.get_head() == self.food:
            self.snake.eat_food()
            self.food = self._generate_food()
            # Comprobar si ha ganado
            if len(self.snake.body) == GRID_WIDTH * GRID_HEIGHT:
                self.game_over = True

    def get_state(self):
        """
        Genera una representación del estado del juego para la red neuronal.
        Devuelve un array plano con: 0=vacío, 1=cuerpo, 2=cabeza, 3=comida.
        """
        state = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for x, y in self.snake.body:
            state[y][x] = 1  # Cuerpo
        head_x, head_y = self.snake.get_head()
        state[head_y][head_x] = 2  # Cabeza
        food_x, food_y = self.food
        state[food_y][food_x] = 3  # Comida
        return state.flatten()

    def get_distance_to_food(self):
        """Calcula la distancia Manhattan entre la cabeza y la comida."""
        head_x, head_y = self.snake.get_head()
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)

    def draw(self):
        """Dibuja el estado actual del juego en la pantalla."""
        if not self.display:
            return

        self.screen.fill(BLACK)
        # Dibujar serpiente
        for i, (x, y) in enumerate(self.snake.body):
            color = GREEN if i == 0 else BLUE
            pygame.draw.rect(self.screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        # Dibujar comida
        food_x, food_y = self.food
        pygame.draw.rect(self.screen, RED, (food_x * GRID_SIZE, food_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        pygame.display.flip()
        self.clock.tick(FPS)

# --- FUNCIONES DE NEAT (ENTRENAMIENTO) CON ESTADÍSTICAS ---

# Variable global para el recolector de estadísticas
stats_collector = NEATStatsCollector()

def eval_genome(genome, config):
    """Evalúa el rendimiento (fitness) de un único genoma."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game(display=False) # Sin pantalla para un entrenamiento rápido

    genome.fitness = 0.0 # Inicializar fitness
    previous_distance = game.get_distance_to_food()

    while not game.game_over:
        state = game.get_state()
        output = net.activate(state)
        action = np.argmax(output) # Elegir la acción con la salida más alta
        game.update(action)

        # Recompensa por acercarse a la comida
        current_distance = game.get_distance_to_food()
        if current_distance < previous_distance:
            genome.fitness += 1.0
        previous_distance = current_distance

    # Fitness final basado en la puntuación y los pasos
    genome.fitness += (game.snake.score * 1000) + game.snake.steps
    if game.snake.score == GRID_WIDTH * GRID_HEIGHT:
        genome.fitness += 50000 # Gran recompensa por ganar

    return genome.fitness

def eval_genomes(genomes, config):
    """Función wrapper para evaluar toda la población de genomas."""
    for genome_id, genome in genomes:
        eval_genome(genome, config)

class CustomStatsReporter(neat.StdOutReporter):
    """Reporter personalizado que recopila estadísticas detalladas."""
    
    def __init__(self, show_species_detail):
        super().__init__(show_species_detail)
        self.generation = 0
    
    def post_evaluate(self, config, population, species, best_genome):
        """Se llama después de cada evaluación de generación."""
        super().post_evaluate(config, population, species, best_genome)
        
        # Recopilar estadísticas para esta generación
        stats_collector.record_generation(
            self.generation, 
            population, 
            species, 
            best_genome
        )
        
        self.generation += 1

def run_neat(config_path):
    """Configura y ejecuta el algoritmo NEAT para entrenar a la IA."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Inicializar recolector de estadísticas
    stats_collector.start_training(config)

    population = neat.Population(config)
    
    # Usar reporter personalizado
    custom_reporter = CustomStatsReporter(True)
    population.add_reporter(custom_reporter)
    
    # Reporter de estadísticas estándar
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    try:
        winner = population.run(eval_genomes, 5000)  # Entrenar por 50 generaciones
        print(f"\nEntrenamiento completado. Mejor genoma: {winner.key}")
        
        # Finalizar recolección de estadísticas
        stats_collector.end_training()

        # Guardar el genoma ganador y la configuración
        script_dir = os.path.dirname(__file__)
        with open(os.path.join(script_dir, 'best_snake_ai.pkl'), 'wb') as f:
            pickle.dump(winner, f)
        with open(os.path.join(script_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        # Generar reporte de Excel
        excel_filename = f"snake_ai_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        stats_collector.generate_excel_report(excel_filename)

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
        stats_collector.end_training()
        # Generar reporte parcial
        if stats_collector.generation_data:
            excel_filename = f"snake_ai_training_report_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            stats_collector.generate_excel_report(excel_filename)
            print(f"Reporte parcial generado: {excel_filename}")

# --- FUNCIONES DE UTILIDAD Y EJECUCIÓN ---

def play_best_ai():
    """Carga y ejecuta la mejor IA guardada en modo visual."""
    script_dir = os.path.dirname(__file__)
    winner_path = os.path.join(script_dir, 'best_snake_ai.pkl')
    config_path = os.path.join(script_dir, 'config.pkl')

    try:
        with open(winner_path, 'rb') as f:
            genome = pickle.load(f)
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos 'best_snake_ai.pkl' o 'config.pkl'.")
        print("Por favor, entrena primero la IA ejecutando el script sin argumentos.")
        return

    print("🐍 Cargando la mejor IA...")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game(display=True) # Con pantalla para jugar

    while not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_over = True
                break
        if game.game_over:
            break

        state = game.get_state()
        output = net.activate(state)
        action = np.argmax(output)
        game.update(action)
        game.draw()

        # Actualizar título de la ventana con la puntuación
        pygame.display.set_caption(f"Snake AI | Puntuación: {game.snake.score}")

    print(f"\nJuego terminado. Puntuación final: {game.snake.score}")
    pygame.quit()

def create_config_file():
    """Crea el archivo de configuración de NEAT en el directorio del script."""
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config.txt')

    num_inputs = GRID_WIDTH * GRID_HEIGHT
    config_content = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100000
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# Network parameters
num_inputs            = {num_inputs}
num_hidden            = 0
num_outputs           = 4
initial_connection    = full_direct
feed_forward          = True

# Compatibility thresholds
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# Mutation probabilities
conn_add_prob         = 0.5
conn_delete_prob      = 0.5
node_add_prob         = 0.2
node_delete_prob      = 0.2

# Activation and aggregation functions
activation_default      = relu
activation_options      = relu sigmoid tanh
activation_mutate_rate  = 0.1
aggregation_default     = sum
aggregation_options     = sum
aggregation_mutate_rate = 0.1

# Bias mutation
bias_init_mean        = 0.0
bias_init_stdev       = 1.0
bias_max_value        = 15.0
bias_min_value        = -15.0
bias_mutate_power     = 0.5
bias_mutate_rate      = 0.7
bias_replace_rate     = 0.1

# Response curve parameters
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 15.0
response_min_value      = -15.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# Weight mutation
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 15
weight_min_value        = -15
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

# Connection enabling parameters
enabled_default         = True
enabled_mutate_rate     = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism              = 2
survival_threshold   = 0.2
"""
    with open(config_path, 'w') as f:
        f.write(config_content)
    return config_path

def install_dependencies():
    """Instala las dependencias necesarias."""
    try:
        import pandas
        import openpyxl
        print("✅ Dependencias ya instaladas")
    except ImportError:
        print("📦 Instalando dependencias necesarias...")
        print("Por favor, ejecuta los siguientes comandos:")
        print("pip install pandas openpyxl")
        print("Luego vuelve a ejecutar el script.")
        sys.exit(1)

def create_advanced_excel_report():
    """Crea un reporte avanzado con gráficos y análisis adicionales."""
    # Esta función podría expandirse para incluir:
    # - Gráficos de evolución del fitness
    # - Análisis de diversidad genética
    # - Comparación entre especies
    # - Predicciones de convergencia
    pass

if __name__ == "__main__":
    # Verificar dependencias
    install_dependencies()
    
    config_file = create_config_file()

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "play":
            play_best_ai()
        elif sys.argv[1].lower() == "report":
            # Opción para generar solo un reporte de datos existentes
            print("🔍 Función de reporte independiente no implementada aún.")
            print("El reporte se genera automáticamente durante el entrenamiento.")
        else:
            print("Uso: python snake_ai.py [play|report]")
    else:
        print("🤖 Iniciando entrenamiento de la IA para Snake con análisis completo...")
        print("📊 Se generará un reporte detallado en Excel al finalizar.")
        print("Presiona Ctrl+C para detener el entrenamiento.")
        print("Para jugar con la mejor IA guardada, ejecuta: python snake_ai.py play")
        time.sleep(2)
        run_neat(config_file)