from gymnasium.envs.registration import register
from quantum_envs.quantum_circuit import QuantumCircuit

register(
    id='quantum_env_xor-v0',
    entry_point='quantum_envs:QuantumCircuit',
)
