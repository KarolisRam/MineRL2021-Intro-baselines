"""
A fully scripted agent on a FIXED WORLD SEED.
It uses the MineRLObtainDiamond environment with env.seed(21).
It gets 6 logs from a nearby tree, makes a wooden_pickaxe and digs down for some cobblestone.
This is a good place to test your scripts.
"""

import gym
import minerl


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
    Specify the action sequence for the agent to execute.
    """
    # get 6 logs:
    action_sequence = []
    action_sequence += [''] * 100  # wait 5 sec
    action_sequence += ['forward'] * 8
    action_sequence += ['attack'] * 61
    action_sequence += ['camera:[-10,0]'] * 7  # look up
    action_sequence += ['attack'] * 61
    action_sequence += ['attack'] * 61
    action_sequence += ['attack'] * 61
    action_sequence += ['attack'] * 61
    action_sequence += [''] * 50
    action_sequence += ['jump']
    action_sequence += ['forward'] * 10
    action_sequence += ['camera:[-10,0]'] * 2
    action_sequence += ['attack'] * 61
    action_sequence += ['attack'] * 61
    action_sequence += ['attack'] * 61
    action_sequence += ['camera:[10,0]'] * 9  # look down
    action_sequence += [''] * 50

    # make planks, sticks, crafting table and wooden pickaxe:
    action_sequence += ['back'] * 2
    action_sequence += ['craft:planks'] * 4
    action_sequence += ['craft:stick'] * 2
    action_sequence += ['craft:crafting_table']
    action_sequence += ['camera:[10,0]'] * 9
    action_sequence += ['jump']
    action_sequence += [''] * 5
    action_sequence += ['place:crafting_table']
    action_sequence += [''] * 10

    # bug: looking straight down at a crafting table doesn't let you craft. So we look up a bit before crafting:
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

    action_sequence = get_action_sequence()

    env.seed(21)
    env.reset()

    for i, action in enumerate(action_sequence):
        obs, reward, done, _ = env.step(str_to_act(env, action))
        if reward > 0:
            print(i, reward)
        if done:
            break

    env.close()


if __name__ == '__main__':
    main()
