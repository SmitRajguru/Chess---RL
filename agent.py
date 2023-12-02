from chessEnv import ChessEnv
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import os
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RLtypes = Enum("RLtypes", ["A2C", "A3C", "PPO"])


# Reinforcement Learning agent for the game of Chess
class ChessBot:
    def __init__(self, env: ChessEnv, enemy):
        self.env = env
        self.enemy = enemy

        state_size = env.observation_space.shape  # (14,8,8)
        action_size = env.action_space.n  # 4096
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.fig = plt.figure(figsize=(16, 8))
        # create 2 axis for  side-by-side plots
        self.ax = [self.fig.add_subplot(1, 2, 1), self.fig.add_subplot(1, 2, 2)]

        self.run_data = {}

        # plt.tight_layout()
        plt.show()

    def render(self):
        self.env.show_board(self.ax[0])

        # show the current epoch, rewards, etc on the second plot
        self.ax[1].cla()
        self.ax[1].axis("off")

        self.ax[1].set_title("Current Run Data")

        i = 1

        # show current run data
        for key, value in self.run_data.items():
            # fit text in plot
            txt = f"{key}: {value}"
            txt = txt.split(",")
            if len(txt) > 1:
                mixText = []
                count = 0
                for text in txt:
                    count += len(text)
                    if count > 80:
                        mixText.append(",".join(txt[: txt.index(text)]))
                        txt = txt[txt.index(text) :]
                        count = 0
                if count != 0:
                    mixText.append(",".join(txt))

                for text in mixText:
                    self.ax[1].text(
                        0,
                        1 - i / 20,
                        text,
                        transform=self.ax[1].transAxes,
                    )
                    i += 1
                i -= 1
            else:
                self.ax[1].text(0, 1 - i / 20, txt[0], transform=self.ax[1].transAxes)

            i += 1

        # update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_action(self, state):
        # # negate value board if player is black
        # if self.env.board.turn == chess.BLACK:
        #     state *= -1
        #     state = np.flipud(state)

        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)
        action = action_probs.sample()
        action = action.cpu().detach().numpy()

        return action, action_probs

    def get_critic(self, state):
        # # negate value board if player is black
        # if self.env.board.turn == chess.BLACK:
        #     state *= -1
        #     state = np.flipud(state)

        state = torch.from_numpy(state).float().to(device)
        value = self.critic(state)
        return value

    def train(self, epochs):
        for epoch in range(epochs):
            done = False
            state = self.env.reset()

            self.render()

            while not done:
                action, action_probs = self.get_action(state.copy())
                next_state, reward, done, info = self.env.step(action)

                # add reward to run_data
                self.run_data["epoch"] = epoch
                self.run_data["reward"] = reward
                self.run_data["action"] = action
                for key, value in info.items():
                    self.run_data[key] = value

                value = self.get_critic(state.copy())
                next_value = self.get_critic(next_state.copy())
                td_target = reward + discount_factor * next_value * (1 - done)
                td_error = td_target - value

                # Actor loss
                log_prob = action_probs.log_prob(torch.tensor(action).to(device))
                actor_loss = -log_prob * td_error.detach()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Critic loss
                critic_loss = td_error**2
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                state = next_state

                self.render()

            if epoch % 10 == 0:
                print("Epoch: {}, Reward: {}".format(epoch, reward))

                if not os.path.exists(f"models/{RLtype.name}/{epoch}"):
                    os.makedirs(f"models/{RLtype.name}/{epoch}")
                torch.save(
                    self.actor.state_dict(), f"models/{RLtype.name}/{epoch}/actor.pth"
                )
                torch.save(
                    self.critic.state_dict(), f"models/{RLtype.name}/{epoch}/critic.pth"
                )


if __name__ == "__main__":
    env = ChessEnv()

    env.mute = True

    lr = 0.0001
    discount_factor = 0.99
    RLtype = RLtypes.A2C

    agent = ChessBot(env, None)
    agent.train(1000000)
