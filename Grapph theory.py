import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import Counter


# ============================================
#   SIMULATION CONFIGURATION
# ============================================
class SimulationConfig:
    def __init__(self):
        # Core parameters
        self.N = 300                     # Total population
        self.simulation_days = 200       # Number of days
        self.initial_infected_count = 3  # Initial infected people

        # Disease parameters
        self.infection_prob = 0.04       # Transmission probability
        self.recovery_time = 12          # Days to recover

        # Vaccination parameters
        self.vaccination_rate = 0.10     # 10%
        self.vaccination_strategy = 'targeted_degree'   # random / none / targeted_degree

        # Network parameters
        self.network_model = 'barabasi_albert'          # erdos_renyi / watts_strogatz / barabasi_albert
        self.p_edge = 0.015
        self.k_neighbors = 6
        self.p_rewire = 0.1
        self.m_edges = 3


# ============================================
#   MAIN SIR SIMULATION CLASS
# ============================================
class SIRSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.G = None
        self.pos = None
        self.history = {'S': [], 'I': [], 'R': []}
        self.peak_infection = (0, 0)

        self.state_colors = {
            'S': '#3498db',
            'I': '#e74c3c',
            'R': '#2ecc71'
        }

        self._setup_simulation()

    # --------------------------
    def _setup_simulation(self):
        print("Setting up simulation...")

        self._create_network()

        for node in self.G.nodes():
            self.G.nodes[node]['state'] = 'S'
            self.G.nodes[node]['infection_time'] = 0

        self._apply_vaccination_strategy()
        self._initialize_infection()

        self.pos = nx.spring_layout(self.G, seed=42)
        print("Setup complete.")

    # --------------------------
    def _create_network(self):
        model = self.config.network_model
        N = self.config.N

        if model == 'erdos_renyi':
            self.G = nx.erdos_renyi_graph(N, self.config.p_edge)

        elif model == 'watts_strogatz':
            self.G = nx.watts_strogatz_graph(N, self.config.k_neighbors, self.config.p_rewire)

        elif model == 'barabasi_albert':
            self.G = nx.barabasi_albert_graph(N, self.config.m_edges)

        else:
            raise ValueError("Unknown network model")

        print(f"Created a {model} network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")

    # --------------------------
    def _apply_vaccination_strategy(self):
        if self.config.vaccination_strategy == 'none':
            return

        num_to_vaccinate = int(self.config.N * self.config.vaccination_rate)

        if self.config.vaccination_strategy == 'random':
            nodes_to_vaccinate = random.sample(list(self.G.nodes()), num_to_vaccinate)

        elif self.config.vaccination_strategy == 'targeted_degree':
            degree_dict = dict(self.G.degree())
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            nodes_to_vaccinate = sorted_nodes[:num_to_vaccinate]

        for node in nodes_to_vaccinate:
            self.G.nodes[node]['state'] = 'R'

        print(f"Vaccinated {len(nodes_to_vaccinate)} people using {self.config.vaccination_strategy} strategy.")

    # --------------------------
    def _initialize_infection(self):
        susceptible = [n for n, d in self.G.nodes(data=True) if d['state'] == 'S']

        initial = random.sample(susceptible, self.config.initial_infected_count)

        for node in initial:
            self.G.nodes[node]['state'] = 'I'

        print(f"Infected {len(initial)} initial nodes.")

    # --------------------------
    def step(self):
        new_infections = []
    
        infected_nodes = [n for n, d in self.G.nodes(data=True) if d['state'] == 'I']

        # Infection
        for node in infected_nodes:
            for neighbor in self.G.neighbors(node):
                if self.G.nodes[neighbor]['state'] == 'S':
                    if random.random() < self.config.infection_prob:
                        new_infections.append(neighbor)

        # Recovery
        newly_recovered = []
        for node in infected_nodes:
            self.G.nodes[node]['infection_time'] += 1
            if self.G.nodes[node]['infection_time'] >= self.config.recovery_time:
                newly_recovered.append(node)
                self.G.nodes[node]['infection_time'] = 0

        # Update states
        for node in new_infections:
            self.G.nodes[node]['state'] = 'I'
        for node in newly_recovered:
            self.G.nodes[node]['state'] = 'R'

        # Track history
        counts = self.get_state_counts()
        self.history['S'].append(counts['S'])
        self.history['I'].append(counts['I'])
        self.history['R'].append(counts['R'])

        # Peak infection
        if counts['I'] > self.peak_infection[1]:
            self.peak_infection = (len(self.history['I']) - 1, counts['I'])

    # --------------------------
    def get_state_counts(self):
        states = [self.G.nodes[n]['state'] for n in self.G.nodes()]
        return Counter(states)

    # --------------------------
    def get_node_colors(self):
        return [self.state_colors[self.G.nodes[n]['state']] for n in self.G.nodes()]


# ============================================
#   VISUALIZATION
# ============================================
def animate_simulation(sim: SIRSimulation):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Disease Spread Simulation", fontsize=20)

    def update(frame):
        sim.step()

        ax1.clear()
        ax1.set_title(f"Day {frame}")
        nx.draw(sim.G, sim.pos, node_color=sim.get_node_colors(), with_labels=False, node_size=40, ax=ax1)

        ax2.clear()
        ax2.set_title("SIR Model Curves")
        days = range(len(sim.history['S']))
        ax2.plot(days, sim.history['S'], label="Susceptible")
        ax2.plot(days, sim.history['I'], label="Infected")
        ax2.plot(days, sim.history['R'], label="Recovered")
        ax2.legend()

    ani = animation.FuncAnimation(fig, update, frames=sim.config.simulation_days, interval=50, repeat=False)
    plt.show()
    return ani


# ============================================
#   FINAL REPORT
# ============================================
def print_final_report(sim: SIRSimulation):
    print("\n========== FINAL REPORT ==========")
    final = sim.get_state_counts()
    print(f"Susceptible: {final['S']}")
    print(f"Infected: {final['I']}")
    print(f"Recovered: {final['R']}")
    print(f"Peak infections: {sim.peak_infection}")
    print("==================================\n")  


# ============================================
#   MAIN
# ============================================
if __name__ == "__main__":
    config = SimulationConfig()
    config.vaccination_strategy = 'targeted_degree'
    config.network_model = 'barabasi_albert'

    print("Starting SIR Simulation...\n")
    simulation = SIRSimulation(config)
    ani = animate_simulation(simulation)
    print_final_report(simulation)
