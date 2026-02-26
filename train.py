from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from src.simulation.env import MultiAgentTrafficEnv, TrafficCallbacks
from src.simulation.runner import SumoRunner
from gymnasium import spaces
import numpy as np

=
runner = SumoRunner("/home/dim8art/traffic-control/data/sumo/lomonosov_mini.net.xml")

config = (
    PPOConfig()
    # Отключаем экспериментальный стек, чтобы избежать ошибок с ID
    .api_stack(
        enable_rl_module_and_learner=False, 
        enable_env_runner_and_connector_v2=False
    )
    .environment(
        MultiAgentTrafficEnv, 
        env_config={"runner": runner, "net_file": runner.net_file}
    )
    .framework("torch")
    .env_runners(num_env_runners=1)
    .callbacks(TrafficCallbacks) 
    .multi_agent(
        policies={
            "traffic_policy": (
                None, 
                spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32), 
                spaces.Discrete(2), 
                {}
            ),
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "traffic_policy",
    )
)

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(stop={"training_iteration": 50})
)
tuner.fit()