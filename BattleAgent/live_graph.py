import matplotlib.pyplot as plt

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

class Graph:
    def __init__(self):
        self.fig = plt.figure(figsize=(6, 3))
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.title('Average Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        self.fig2 = plt.figure(figsize=(6, 3))
        self.ax2 = self.fig2.add_subplot(1, 1, 1)
        plt.title('Real-Time Rewards')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        self.fig3 = plt.figure(figsize=(6, 3))
        self.ax3 = self.fig3.add_subplot(1, 1, 1)
        plt.title('Winning Rate in Every 10 Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Winning Rate')
        plt.ion()
        plt.pause(0.001)

    def animate(self, xs, ys):
        self.ax.clear()
        self.ax.plot(xs, ys)
        self.ax.set_title('Average Reward per Episode')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Average Reward')
        self.fig.savefig('Average Reward per Episode.png')
        plt.pause(0.001)

    def animate_episode(self, xs, ys):
        self.ax2.clear()
        self.ax2.plot(xs, ys)
        self.ax2.set_title('Real-Time Rewards')
        self.ax2.set_xlabel('Timestep')
        self.ax2.set_ylabel('Reward')
        self.fig2.savefig('Real-Time Rewards.png')
        plt.pause(0.001)
    
    def animate_wins(self, xs, ys):
        self.ax3.clear()
        self.ax3.plot(xs, ys)
        self.ax3.set_title('Winning Rate in Every 10 Episodes')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Winning Rate')
        self.fig3.savefig('Winning Rate in Every 10 Episodes.png')
        plt.pause(0.001)

    def clear_episode(self):
        self.ax2.clear()
        plt.pause(0.001)
