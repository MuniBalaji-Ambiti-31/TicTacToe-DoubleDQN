from TicTacToe import TicTacToe
from Network import Network
from ReplayMemory import Transition, ReplayMemory
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import random
import logging
import io

from ReplayMemory import ReplayMemory

def train(
    n_steps: int = 100_000,
    batch_size: int = 128,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_steps: int = 200_000,
) -> bytes:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Beginning training on: {}".format(device))
    logging.info("Beginning training on: {}".format(device))
    target_update = int((1e-2) * n_steps)
    policy = Network(n_inputs=3 * 9, n_outputs=9).to(device)
    target = Network(n_inputs=3 * 9, n_outputs=9).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    memory = ReplayMemory(50_000)
    env = TicTacToe()
    state = torch.tensor(np.array([env.reset()]), dtype=torch.float).to(device)
    old_summary = {
        "total games": 0,
        "ties": 0,
        "illegal moves": 0,
        "player 0 wins": 0,
        "player 1 wins": 0,
    }
    _randoms = 0
    summaries = []
    
    for step in range(n_steps):
        t = np.clip(step / eps_steps, 0, 1)
        eps = (1 - t) * eps_start + t * eps_end
        
        action, was_random = select_model_action(device, policy, state, eps)
        if was_random:
            _randoms += 1
        next_state, reward, done, _ = env.step(action.item())
        
        if not done:
            next_state, _, done, _ = env.step(select_dummy_action(next_state))
            next_state = torch.tensor(np.array([next_state]), dtype=torch.float).to(device)
        
        if done:
            next_state = None
        
        memory.push(state, action, next_state, torch.tensor([reward], device=device))
        state = next_state
        
        optimize_model(
            device=device,
            optimizer=optimizer,
            policy=policy,
            target=target,
            memory=memory,
            batch_size=batch_size,
            gamma=gamma,
        )
        
        if done:
            state = torch.tensor(np.array([env.reset()]), dtype=torch.float).to(device)
        
        if step % target_update == 0:
            target.load_state_dict(policy.state_dict())
        
        if step % 5000 == 0:
            delta_summary = {k: env.summary[k] - old_summary[k] for k in env.summary}
            delta_summary["random actions"] = _randoms
            old_summary = {k: env.summary[k] for k in env.summary}
            logging.info("{} : {}".format(step, delta_summary))
            print("{} : {}".format(step, delta_summary))
            summaries.append(delta_summary)
            _randoms = 0
    
    logging.info("Complete")
    
    res = io.BytesIO()
    torch.save(policy.state_dict(), res)
    print("----------saving model in file-----------------")
    checkpoint_data = {
    'epoch': n_steps,
    'state_dict': policy.state_dict(),
    }
    ckpt_path = os.path.join("checkpoint/tictactoe_policy_model.pt")
    torch.save(checkpoint_data, ckpt_path)
    
    return res.getbuffer()

def optimize_model(
    device: torch.device,
    optimizer: optim.Optimizer,
    policy: Network,
    target: Network,
    memory: ReplayMemory,
    batch_size: int,
    gamma: float,
):
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    
    target_state_action_values = (next_state_values * gamma) + reward_batch
    
    loss = F.smooth_l1_loss(
        state_action_values,
        target_state_action_values.unsqueeze(1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def select_dummy_action(state: np.array) -> int:
    state = state.reshape(3, 3, 3)
    open_spots = state[:, :, 0].reshape(-1)
    p = open_spots / open_spots.sum()
    return np.random.choice(np.arange(9), p=p)

def select_model_action(
    device: torch.device,
    model: Network,
    state: torch.tensor,
    eps: float
) -> Tuple[torch.tensor, bool]:
    sample = random.random()
    if sample > eps:
        return model.act(state), False
    else:
        return (
            torch.tensor(
                [[random.randrange(0, 9)]],
                device=device,
                dtype=torch.long,
            ),
            True,
        )
