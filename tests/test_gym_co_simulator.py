import pytest
import numpy as np
from cosimulator import GymCoSimulator
from cosimulator.network import NS3NetworkSimulator

def test_gym_co_simulator_initialization():
    """Test initialization of Gym co-simulator."""
    network_sim = NS3NetworkSimulator()
    env = gym.make('CartPole-v1')
    co_sim = GymCoSimulator(network_sim, env)
    
    assert co_sim.network_simulator == network_sim
    assert co_sim.robotics_simulator == env

def test_gym_co_simulator_step():
    """Test step function of Gym co-simulator."""
    network_sim = NS3NetworkSimulator()
    env = gym.make('CartPole-v1')
    co_sim = GymCoSimulator(network_sim, env)
    
    observation = co_sim.reset()
    action = env.action_space.sample()
    observation, reward, done, info = co_sim.step(action)
    
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_gym_co_simulator_reset():
    """Test reset function of Gym co-simulator."""
    network_sim = NS3NetworkSimulator()
    env = gym.make('CartPole-v1')
    co_sim = GymCoSimulator(network_sim, env)
    
    observation = co_sim.reset()
    assert isinstance(observation, np.ndarray)
    assert co_sim.scheduled_messages == {} 