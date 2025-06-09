import time
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from app.agent import QLearningAgent


class QGrid:
    def __init__(self):
        self.cell_size = 50
        self.grid = [
            [1, 1, 1, 1, -100, 1, 1, 1, 1, 1, 1, -100, None],
            [1, -100, 1, 1, 1, 1, 1, 1, 1, 1, 1, -100, None],
            [-100, 1, -100, 1, 1, 1, 1, 1, -100, -100, 1, 1, None],
            [1, -100, 1, 1, 1, 1, 1, 1, -100, 1, 1, 100, None],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None],
            [None, None, None, None, 1, 1, -100, 1, None, None, None, None, None],
            [None, None, None, None, 1, 1, 1, 1, None, None, None, None, None],
            [None, None, None, None, 1, 1, 1, 1, None, None, None, None, None],
            [None, None, None, None, 1, 1, -100, 1, None, None, None, None, None],
            [None, None, None, None, 1, 1, 1, 1, None, None, None, None, None],
        ]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start_pos = (9, 4)
        self.goal_pos = (3, 11)
        self.get_initial_params()

    def get_initial_params(self):
        self.root = tk.Tk()
        self.root.title("Q-Learning Configuração")
        self.config_frame = ttk.Frame(self.root, padding="20")
        self.config_frame.pack()
        ttk.Label(self.config_frame, text="Parâmetros do Q-Learning").grid(row=0, columnspan=2, pady=10)
        ttk.Label(self.config_frame, text="Epsilon (exploração):").grid(row=1, column=0, sticky="e")
        self.epsilon_var = tk.DoubleVar(value=0.3)
        self.epsilon_entry = ttk.Entry(self.config_frame, textvariable=self.epsilon_var, width=10)
        self.epsilon_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(self.config_frame, text="Alpha (aprendizado):").grid(row=2, column=0, sticky="e")
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.alpha_entry = ttk.Entry(self.config_frame, textvariable=self.alpha_var, width=10)
        self.alpha_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(self.config_frame, text="Gamma (desconto):").grid(row=3, column=0, sticky="e")
        self.gamma_var = tk.DoubleVar(value=0.9)
        self.gamma_entry = ttk.Entry(self.config_frame, textvariable=self.gamma_var, width=10)
        self.gamma_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.start_button = ttk.Button(self.config_frame, text="Iniciar Simulação", command=self.start_simulation)
        self.start_button.grid(row=4, columnspan=2, pady=20)
        self.canvas = None
        self.circle = None

    def start_simulation(self):
        self.config_frame.destroy()

        self.root.title("Q-Learning")
        self.canvas = tk.Canvas(self.root, width=self.cols * self.cell_size, height=self.rows * self.cell_size)
        self.canvas.pack()

        self.draw_grid()
        self.circle = self.draw_agent(self.start_pos)

        self.agent = QLearningAgent(
            epsilon=self.epsilon_var.get(), alpha=self.alpha_var.get(), gamma=self.gamma_var.get()
        )

        self.root.after(1, self.run_training(self.agent))

    def get_reward(self, state):
        i, j = state
        return self.grid[i][j]

    def get_valid_actions(self, state):
        i, j = state
        actions = []
        if i > 0 and self.grid[i - 1][j] is not None:
            actions.append('N')
        if i < self.rows - 1 and self.grid[i + 1][j] is not None:
            actions.append('S')
        if j < self.cols - 1 and self.grid[i][j + 1] is not None:
            actions.append('E')
        if j > 0 and self.grid[i][j - 1] is not None:
            actions.append('W')
        return actions

    def next_state(self, state, action):
        i, j = state
        if action == 'N':
            return (i - 1, j)
        if action == 'S':
            return (i + 1, j)
        if action == 'E':
            return (i, j + 1)
        if action == 'W':
            return (i, j - 1)
        return state

    def draw_grid(self):
        for i in range(self.rows):
            for j in range(self.cols):
                reward = self.grid[i][j]
                if reward is None:
                    continue

                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                if reward == -100:
                    color = "black"
                elif reward == 100:
                    color = "green"
                elif (i, j) == self.start_pos:
                    color = "red"
                else:
                    color = "white"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=str(reward), fill="purple")

    def draw_agent(self, pos):
        x1 = pos[1] * self.cell_size + 10
        y1 = pos[0] * self.cell_size + 10
        x2 = pos[1] * self.cell_size + self.cell_size - 10
        y2 = pos[0] * self.cell_size + self.cell_size - 10
        return self.canvas.create_oval(x1, y1, x2, y2, fill="blue")

    def move_agent(self, pos):
        x1 = pos[1] * self.cell_size + 10
        y1 = pos[0] * self.cell_size + 10
        x2 = pos[1] * self.cell_size + self.cell_size - 10
        y2 = pos[0] * self.cell_size + self.cell_size - 10
        self.canvas.coords(self.circle, x1, y1, x2, y2)

    def draw_policy_arrows(self, agent):
        self.canvas.delete("policy_arrow")
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] is None:
                    continue
                if (i, j) == self.goal_pos:
                    continue

                state = (i, j)
                valid_actions = self.get_valid_actions(state)
                best_action = agent.get_policy(state, valid_actions)
                if best_action is None:
                    continue

                center_x = j * self.cell_size + self.cell_size // 2
                center_y = i * self.cell_size + self.cell_size // 2
                length = 15

                if best_action == 'N':
                    self.canvas.create_line(
                        center_x,
                        center_y,
                        center_x,
                        center_y - length,
                        arrow=tk.LAST,
                        fill="blue",
                        tags="policy_arrow",
                    )
                elif best_action == 'S':
                    self.canvas.create_line(
                        center_x,
                        center_y,
                        center_x,
                        center_y + length,
                        arrow=tk.LAST,
                        fill="blue",
                        tags="policy_arrow",
                    )
                elif best_action == 'E':
                    self.canvas.create_line(
                        center_x,
                        center_y,
                        center_x + length,
                        center_y,
                        arrow=tk.LAST,
                        fill="blue",
                        tags="policy_arrow",
                    )
                elif best_action == 'W':
                    self.canvas.create_line(
                        center_x,
                        center_y,
                        center_x - length,
                        center_y,
                        arrow=tk.LAST,
                        fill="blue",
                        tags="policy_arrow",
                    )

    def update_display(self, agent, delay=0):
        self.draw_policy_arrows(agent)
        self.root.update()
        if delay > 0:
            time.sleep(delay)

    def run_episode(self, agent, start_time, training=True, episode_num=None):
        state = self.start_pos
        steps = 0
        self.move_agent(state)
        while state != self.goal_pos:
            valid_actions = self.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions, training)
            next_state = self.next_state(state, action)
            reward = self.get_reward(next_state)
            next_valid_actions = self.get_valid_actions(next_state)

            if training:
                agent.learn(state, action, reward, next_state, next_valid_actions)

            self.move_agent(next_state)
            state = next_state
            steps += 1
            delay = 1
            if training:
                delay = 0
            self.update_display(agent, delay)

        if episode_num is not None:
            time_spend = time.time() - start_time
            print(f"Tempo: {time_spend:.2f} segundos")
            print(f"Episódio {episode_num} finalizado em {steps} passos")

    def run_training(self, agent, episodes=200):
        start_time = time.time()

        for ep in range(episodes):
            self.run_episode(agent, start_time=start_time, training=True, episode_num=ep + 1)
        print("Treinamento concluído!")

        while True:
            self.run_episode(agent, start_time=start_time, training=False)
            answer = simpledialog.askinteger(
                "Continuar",
                "Mostrar a política ótima de novo? (1-sim 0-não)",
                parent=self.root,
                minvalue=0,
                maxvalue=1,
            )

            if answer is None or answer == 0:
                break

        self.root.quit()

    def start(self):
        self.root.mainloop()
