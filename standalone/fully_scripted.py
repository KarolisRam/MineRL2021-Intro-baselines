"""
A fully scripted agent that doesn't require training. It has two parts:
1. Bulldozer the lumberjack - a script that simply digs forward with occasional jumps and random 90 degree turns.
2. A script that crafts a wooden pickaxe and digs down to get some cobblestone.
Part one runs until a certain number of logs is achieved, then part 2 kicks in.
When evaluated on MineRLObtainDiamond environment it achieves an average reward of 4.0.
"""

import random
import gym
import minerl


# Parameters:
TEST_EPISODES = 10  # number of episodes to test the agent for.
MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamond.
N_WOOD_THRESHOLD = 4  # number of wood logs to get before starting script #2.


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


def get_action_sequence_bulldozer():
    """
    Specify the action sequence for Bulldozer, the scripted lumberjack.
    """
    action_sequence_bulldozer = []
    action_sequence_bulldozer += [''] * 100  # wait 5 secs
    action_sequence_bulldozer += ['camera:[10,0]'] * 3  # look down 30 degrees

    for _ in range(100):
        action_sequence_bulldozer += ['attack sprint forward'] * 100  # dig forward for 5 secs
        action_sequence_bulldozer += ['jump']  # jump!
        action_sequence_bulldozer += ['attack sprint forward'] * 100
        action_sequence_bulldozer += ['jump']
        action_sequence_bulldozer += ['attack sprint forward'] * 100
        if random.random() < 0.5:  # turn either 90 degrees left or 90 degrees right with an equal probability
            action_sequence_bulldozer += ['camera:[0,-10]'] * 9
        else:
            action_sequence_bulldozer += ['camera:[0,10]'] * 9
    return action_sequence_bulldozer


def get_action_sequence():
    """
    Specify the action sequence for the script #2.
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


def main():
    env = gym.make('MineRLObtainDiamond-v0')

    # optional interactive mode, where you can connect to your agent and play together (see link for details):
    # https://minerl.io/docs/tutorials/minerl_tools.html#interactive-mode-minerl-interactor
    # env.make_interactive(port=6666, realtime=True)

    for episode in range(TEST_EPISODES):
        env.reset()
        done = False
        total_reward = 0
        steps = 0

        action_sequence_bulldozer = get_action_sequence_bulldozer()
        action_sequence = get_action_sequence()

        # scripted part to get some logs:
        for j, action in enumerate(action_sequence_bulldozer[:MAX_TEST_EPISODE_LEN]):
            obs, reward, done, _ = env.step(str_to_act(env, action))
            total_reward += reward
            steps += 1
            if obs['inventory']['log'] >= N_WOOD_THRESHOLD:
                break
            if done:
                break

        # scripted part to use the logs:
        if not done:
            for i, action in enumerate(action_sequence[:MAX_TEST_EPISODE_LEN - j]):
                obs, reward, done, _ = env.step(str_to_act(env, action))
                total_reward += reward
                steps += 1
                if done:
                    break

        print(f'Episode #{episode+1} reward: {total_reward}\t\t episode length: {steps}')

    env.close()


if __name__ == '__main__':
    main()
