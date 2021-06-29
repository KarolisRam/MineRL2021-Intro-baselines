"""
Reinforcement Learning agent that trains on MineRLTreechop environment. It is then evaluated on MineRLObtainDiamond by
running it for a certain number of ticks and then switching to the scripted part that crafts a wooden_pickaxe and digs
down to get some cobblestone.
With default parameters it trains in about 8 hours on a machine with a GeForce RTX 2080 Ti GPU.
It uses less than 8GB RAM and achieves an average reward of 8.3.
"""

import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
import minerl  # it's important to import minerl after SB3, otherwise model.save doesn't work...


# Parameters:
config = {
    "TRAIN_TIMESTEPS": 2000000,  # number of steps to train the agent for. At 70 FPS 2m steps take about 8 hours.
    "TRAIN_ENV": 'MineRLTreechop-v0',  # training environment for the RL agent. Could use MineRLObtainDiamondDense-v0 here.
    "TRAIN_MODEL_NAME": 'potato',  # name to use when saving the trained agent.
    "TEST_MODEL_NAME": 'potato',  # name to use when loading the trained agent.
    "TEST_EPISODES": 10,  # number of episodes to test the agent for.
    "MAX_TEST_EPISODE_LEN": 18000,  # 18k is the default for MineRLObtainDiamond.
    "TREECHOP_STEPS": 2000,  # number of steps to run RL lumberjack for in evaluations.
}
experiment_name = f"ppo_{int(time.time())}"


def make_env(idx):
    def thunk():
        env = gym.make(config["TRAIN_ENV"])
        env = PovOnlyObservation(env)
        env = ActionShaping(env, always_attack=True)
        env = gym.wrappers.RecordEpisodeStatistics(env) # record stats such as returns
        if idx == 0:
            env = gym.wrappers.Monitor(env, f"videos/{experiment_name}") # record videos
        return env
    return thunk


def track_exp(project_name=None):
    import wandb
    wandb.init(
        project=project_name,
        config=config,
        sync_tensorboard=True,
        name=experiment_name,
        monitor_gym=True,
        save_code=True,
    )

class PovOnlyObservation(gym.ObservationWrapper):
    """
    Turns the observation space into POV only, ignoring the inventory. This is needed for stable_baselines3 RL agents,
    as they don't yet support dict observations. The support should be coming soon (as of April 2021).
    See following PR for details:
    https://github.com/DLR-RM/stable-baselines3/pull/243
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']


class ActionShaping(gym.ActionWrapper):
    """
    The default MineRL action space is the following dict:

    Dict(attack:Discrete(2),
         back:Discrete(2),
         camera:Box(low=-180.0, high=180.0, shape=(2,)),
         craft:Enum(crafting_table,none,planks,stick,torch),
         equip:Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         forward:Discrete(2),
         jump:Discrete(2),
         left:Discrete(2),
         nearbyCraft:Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         nearbySmelt:Enum(coal,iron_ingot,none),
         place:Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch),
         right:Discrete(2),
         sneak:Discrete(2),
         sprint:Discrete(2))

    It can be viewed as:
         - buttons, like attack, back, forward, sprint that are either pressed or not.
         - mouse, i.e. the continuous camera action in degrees. The two values are pitch (up/down), where up is
           negative, down is positive, and yaw (left/right), where left is negative, right is positive.
         - craft/equip/place actions for items specified above.
    So an example action could be sprint + forward + jump + attack + turn camera, all in one action.

    This wrapper makes the action space much smaller by selecting a few common actions and making the camera actions
    discrete. You can change these actions by changing self._actions below. That should just work with the RL agent,
    but would require some further tinkering below with the BC one.
    """
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            # [('back', 1)],
            # [('left', 1)],
            # [('right', 1)],
            # [('jump', 1)],
            # [('forward', 1), ('attack', 1)],
            # [('craft', 'planks')],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def train():
    env = DummyVecEnv([make_env(i) for i in range(1)])
    # For all the PPO hyperparameters you could tune see this:
    # https://github.com/DLR-RM/stable-baselines3/blob/6f822b9ed7d6e8f57e5a58059923a5b24e8db283/stable_baselines3/ppo/ppo.py#L16
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=f"runs/{experiment_name}")
    model.learn(total_timesteps=config["TRAIN_TIMESTEPS"])  # 2m steps is about 8h at 70 FPS
    model.save(config["TRAIN_MODEL_NAME"])

    env.close()


def str_to_act(env, actions):
    """
    Simplifies specifying actions for the scripted part of the agent.
    Some examples for a string with a single action:
        'craft:planks'
        'camera:[10,0]'
        'attack'
        'jump'
        ''
    There should be no spaces in single actions, as we use spaces to separate actions with multiple "buttons" pressed:
        'attack sprint forward'
        'forward camera:[0,10]'

    :param env: base MineRL environment.
    :param actions: string of actions.
    :return: dict action, compatible with the base MineRL environment.
    """
    act = env.action_space.noop()
    for action in actions.split():
        if ":" in action:
            k, v = action.split(':')
            if k == 'camera':
                act[k] = eval(v)
            else:
                act[k] = v
        else:
            act[action] = 1
    return act


def get_action_sequence():
    """
    Specify the action sequence for the scripted part of the agent.
    """
    # make planks, sticks, crafting table and wooden pickaxe:
    action_sequence = []
    action_sequence += [''] * 100
    action_sequence += ['craft:planks'] * 4
    action_sequence += ['craft:stick'] * 2
    action_sequence += ['craft:crafting_table']
    action_sequence += ['camera:[10,0]'] * 18
    action_sequence += ['attack'] * 20
    action_sequence += [''] * 10
    action_sequence += ['jump']
    action_sequence += [''] * 5
    action_sequence += ['place:crafting_table']
    action_sequence += [''] * 10

    # bug: looking straight down at a crafting table doesn't let you craft. So we look up a bit before crafting.
    action_sequence += ['camera:[-1,0]']
    action_sequence += ['nearbyCraft:wooden_pickaxe']
    action_sequence += ['camera:[1,0]']
    action_sequence += [''] * 10
    action_sequence += ['equip:wooden_pickaxe']
    action_sequence += [''] * 10

    # dig down:
    action_sequence += ['attack'] * 600
    action_sequence += [''] * 10

    return action_sequence


def test():
    writer = SummaryWriter(f"runs/{experiment_name}")
    env = gym.make('MineRLObtainDiamond-v0')

    # optional interactive mode, where you can connect to your agent and play together (see link for details):
    # https://minerl.io/docs/tutorials/minerl_tools.html#interactive-mode-minerl-interactor
    # env.make_interactive(port=6666, realtime=True)

    env = PovOnlyObservation(env)
    env = ActionShaping(env, always_attack=True)
    env = gym.wrappers.Monitor(env, f"videos/{experiment_name}")
    env1 = env.unwrapped

    model = PPO.load(config["TEST_MODEL_NAME"], verbose=1)
    model.set_env(env)

    action_sequence = get_action_sequence()

    for episode in range(config["TEST_EPISODES"]):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # RL part to get some logs:
        for i in range(config["TREECHOP_STEPS"]):
            action = model.predict(obs)
            obs, reward, done, _ = env.step(action[0])
            total_reward += reward
            steps += 1
            if done:
                break

        # scripted part to use the logs:
        if not done:
            for i, action in enumerate(action_sequence[:config["MAX_TEST_EPISODE_LEN"] - config["TREECHOP_STEPS"]]):
                obs, reward, done, _ = env1.step(str_to_act(env1, action))
                total_reward += reward
                steps += 1
                if done:
                    break

        print(f'Episode #{episode + 1} return: {total_reward}\t\t episode length: {steps}')
        writer.add_scalar("return", total_reward, global_step=episode)

    env.close()


def main():
    # uncomment the following to upload the logs and videos to Weights and Biases
    track_exp(project_name="minerl")

    # train()
    test()


if __name__ == '__main__':
    main()
