import yaml

algo_config = yaml.load(open("config/algo_config.yaml", "r"), Loader=yaml.FullLoader)
env_config = yaml.load(open("config/env_config.yaml", "r"), Loader=yaml.FullLoader)

CONFIG = {
    "algo": algo_config,
    "env": env_config
}

value_based_algos = ["dqn", "d2qn", "d3qn", "per", "c51", "rainbow"]
policy_based_algos = []