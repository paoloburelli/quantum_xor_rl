import sys
import gymnasium as gym
import os
from typing import Any, Dict, Tuple, Union

import mlflow
import numpy as np
from datetime import datetime

from sb3_contrib import MaskablePPO
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement, \
    EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import quantum_envs


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
            self,
            key_values: Dict[str, Any],
            key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
            step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
                sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
execution_id = os.environ.get('SLURM_JOB_ID', "local")
envs = int(os.environ.get('SLURM_CPUS_PER_TASK', "8"))
cardinality = int(os.environ.get('MATRIX_CARDINALITY', "4"))
env_name = "quantum_env_xor-v0"

run_name = f"{env_name}-action_mask-{cardinality}x{cardinality}-{envs}-{timestamp}-{execution_id}"

if __name__ == '__main__':
    eval_env = make_vec_env(env_name, n_envs=envs, vec_env_cls=SubprocVecEnv,
                            env_kwargs={"cardinality": cardinality})
    train_env = make_vec_env(env_name, n_envs=envs, vec_env_cls=SubprocVecEnv,
                             env_kwargs={"cardinality": cardinality})

    reward_stop = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    stagnation_stop = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=100, verbose=1)

    eval_callback = MaskableEvalCallback(eval_env, eval_freq=10000, callback_after_eval=stagnation_stop,
                                         callback_on_new_best=reward_stop,
                                         n_eval_episodes=100, best_model_save_path=f"./models/{run_name}/",
                                         verbose=1)

    mlflow.set_experiment("quantum_rl")
    with mlflow.start_run(run_name=run_name):
        model = MaskablePPO(policy=MaskableActorCriticPolicy, env=train_env, verbose=0)
        model.set_logger(loggers)
        model.learn(total_timesteps=100000000, log_interval=1, callback=eval_callback)
