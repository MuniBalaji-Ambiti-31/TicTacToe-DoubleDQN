import os
import time
import io
import torch
import torch.nn.functional as F
import sys
from train import train
from TicTacToe import TicTacToe
from Network import Network
import numpy as np

def play(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe()
    done = False
    obs = env.reset()
    exp = {}
    player = 0
    
    while not done:
        time.sleep(1)
        print("Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(*[x for x in range(0, 9)]))
        env.render()  # display current state
        action = None
        
        if player == 1:
            action = int(input())  # human player's input for cell number
        else:
            time.sleep(1)  # trained network's decision
            action = act(model, torch.tensor(np.array([obs]), dtype=torch.float).to(device)).item()
        
        obs, _, done, exp = env.step(action)
        player = 1 - player  # change player turn from 0 to 1 or 1 to 0
        
    #os.system("cls")
    print("Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(*[x for x in range(0, 9)]))
    env.render()
    print(exp)
        
    if "tied" in exp["reason"]:
        print("A tied game. ---------.")
        exit(0)

def load_model(path: str, device: torch.device):  # load trained model
    model = Network(n_inputs=3*9, n_outputs=9).to(device)
    encodings = ['utf-8', 'latin1']
    for encoding in encodings:
        try:
            checkpoint_data = torch.load(path, map_location=device, encoding=encoding)
            model.load_state_dict(checkpoint_data['state_dict'])
            model.eval()
            return model
        except Exception as e:
            print(f"An error occurred while loading the model checkpoint with encoding '{encoding}':", e)
            continue
    print("Failed to load the model checkpoint with any of the specified encodings.")
    sys.exit(1)

# get action from trained network
def act(model: Network, state: torch.tensor):
    with torch.no_grad():
        p = F.softmax(model.forward(state), dim=-1).cpu().numpy()  # 9 outputs from network
        # find non-empty cells
        # following produces 9 values with True or False, False for non-empty cell
        valid_moves = (state.cpu().numpy().reshape(3, 3, 3).argmax(axis=2).reshape(-1) == 0)
        p = valid_moves * p  # 9 numbers with 0 where the cell is not empty
        return p.argmax()  # best decision (0-8) on non-empty cells

def main():
    #res = train() # -------uncomment this line to train network
    # after model is trained, use the following code to play
    # human is player 2, select the cell number 0-8 to decide which cell
    # you want for your turn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(r'checkpoint\tictactoe_policy_model.pt', device)  # Using raw string literal
    play(model)  # play one game against trained network

if __name__ == "__main__":
    sys.exit(int(main() or 0))




















