# TicTacToeDoubleDQN

## 🎯 Project Overview
This repository implements a Double Deep Q-Network (Double DQN) agent for playing TicTacToe using PyTorch and Gymnasium. The agent learns to avoid illegal moves and outperforms a random opponent.

## ⚙️ Setup & Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/tic-tac-toe-double-dqn-pytorch.git
   cd TicTacToeDoubleDQN
   ```
2. **Create & activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage
- **Train the agent**
  ```bash
  python train.py --steps 100000 --batch-size 128
  ```
- **Play against the trained model**
  ```bash
  python TicTacToeDoubleDQN.py --play --model-path checkpoint/tictactoe_policy_model.pt
  ```

## 📁 File Structure
```
TicTacToeDoubleDQN/
├── Network.py              # Neural network definition
├── TicTacToe.py            # Gym environment for TicTacToe
├── ReplayMemory.py         # Experience replay buffer
├── train.py                # Training script
├── TicTacToeDoubleDQN.py   # Entry point for training and play
├── requirements.txt        # Python dependencies
├── checkpoint/             # Model checkpoints (.pt)
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

## ⚖️ License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
