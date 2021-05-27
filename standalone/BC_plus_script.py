"""
Behavioural Cloning agent that trains on MineRLTreechop data. It is then evaluated on MineRLObtainDiamond by running it
for a certain number of steps and then switching to the scripted part that crafts a wooden_pickaxe and digs down to get
some cobblestone.
With default parameters it trains in 5-10 mins on a machine with a GeForce RTX 2080 Ti GPU.
It uses less than 8GB RAM and achieves an average reward of 8.6.
"""

from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import gym
import minerl


# Parameters:
DATA_DIR = "data"  # path to where MineRL dataset resides (should contain "MineRLTreechop-v0" directory).
EPOCHS = 3  # How many times we train over the dataset.
LEARNING_RATE = 0.0001  # Learning rate for the neural network.

TRAIN_MODEL_NAME = 'another_potato.pth'  # name to use when saving the trained agent.
TEST_MODEL_NAME = 'another_potato.pth'  # name to use when loading the trained agent.

TEST_EPISODES = 10  # number of episodes to test the agent for.
MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamond.
TREECHOP_STEPS = 2000  # number of steps to run BC lumberjack for in evaluations.


class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


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


def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size,), dtype=np.int)

    for i in range(len(camera_actions)):
        # Moving camera is most important (horizontal first)
        if camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = -1
    return actions


def train():
    data = minerl.data.make("MineRLTreechop-v0",  data_dir=DATA_DIR, num_workers=4)

    # We know ActionShaping has seven discrete actions, so we create
    # a network to map images to seven values (logits), which represent
    # likelihoods of selecting those actions
    network = NatureCNN((3, 64, 64), 7).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(data.batch_iter(num_epochs=EPOCHS, batch_size=32, seq_len=1)):
        # We only use pov observations (also remove dummy dimensions)
        obs = dataset_obs["pov"].squeeze().astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations
        obs /= 255.0

        # Actions need bit more work
        actions = dataset_action_batch_to_actions(dataset_actions)

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

        # Obtain logits of each action
        logits = network(th.from_numpy(obs).float().cuda())

        # Minimize cross-entropy with target labels.
        # We could also compute the probability of demonstration actions and
        # maximize them.
        loss = loss_function(logits, th.from_numpy(actions).long().cuda())

        # Standard PyTorch update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()

    th.save(network, TRAIN_MODEL_NAME)
    del data


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
    network = th.load(TEST_MODEL_NAME).cuda()

    env = gym.make('MineRLObtainDiamond-v0')

    # optional interactive mode, where you can connect to your agent and play together (see link for details):
    # https://minerl.io/docs/tutorials/minerl_tools.html#interactive-mode-minerl-interactor
    # env.make_interactive(port=6666, realtime=True)

    env = ActionShaping(env, always_attack=True)
    env1 = env.unwrapped

    num_actions = env.action_space.n
    action_list = np.arange(num_actions)

    action_sequence = get_action_sequence()

    for episode in range(TEST_EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # BC part to get some logs:
        for i in range(TREECHOP_STEPS):
            # Process the action:
            #   - Add/remove batch dimensions
            #   - Transpose image (needs to be channels-last)
            #   - Normalize image
            obs = th.from_numpy(obs['pov'].transpose(2, 0, 1)[None].astype(np.float32) / 255).cuda()
            # Turn logits into probabilities
            probabilities = th.softmax(network(obs), dim=1)[0]
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()
            # Sample action according to the probabilities
            action = np.random.choice(action_list, p=probabilities)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break

        # scripted part to use the logs:
        if not done:
            for i, action in enumerate(action_sequence[:MAX_TEST_EPISODE_LEN - TREECHOP_STEPS]):
                obs, reward, done, _ = env1.step(str_to_act(env1, action))
                total_reward += reward
                steps += 1
                if done:
                    break

        print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}')

    env.close()


def main():
    # train()
    test()


if __name__ == '__main__':
    main()
