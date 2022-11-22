from dodger import Dodger			# dodger ai
try:
    from malmo import MalmoPython
except:
    import MalmoPython				# Malmo
import logging						# world debug prints
import time							# sleep for a few ticks every trial
import os							# os.system.clear()
import random						# random start delay
import world						# world manipulation/observation functions
import pickle
import matplotlib.pyplot as plt
import sys 
from collections import deque
import numpy
from dodger import Dodger
from main import draw_success_rate
logging.basicConfig(level=logging.DEBUG)
mission_xml = "mission6.xml"
test_epsilon = 0.1
# ========================================================
# some func here
# ========================================================
def test_getAction(dodger, state, possible_actions):
    if state not in dodger.q_table:
        dodger.q_table[state] = {}
        for a in possible_actions:
            dodger.q_table[state][a] = 0
    
    if random.random() < test_epsilon:
        action_i = random.randint(0, len(possible_actions) - 1)
    else:
        values = [dodger.q_table[state][a] for a in possible_actions]
        action_i = numpy.argmax(values)
    return possible_actions[action_i]


# ========================================================
# test
# ========================================================


def test_run(dodger):
    # move or not
    possible_actions = ['move 1', 'move 0']

    # returns total reward and success flag
    total_reward, success = 0, None

    # inititalize terminating state
    terminate_s = 'ENDDING'
    dodger.q_table[terminate_s] = {}
    for a in possible_actions:
        dodger.q_table[terminate_s][a] = 0

    # run until damaged
    running = True



    while running:
        # get initial state/action/reward
        obs = world.get_observations(dodger.agent_host)
        s0 = dodger.get_curr_state(obs)
        a = a0 = test_getAction(dodger, s0, possible_actions)
        r0 = 0

        # continuously get observations
        T = sys.maxsize
        for t in range(sys.maxsize):
            obs = world.get_observations(dodger.agent_host)
            curr_pos = world.get_curr_pos(obs)
            dodger.life = world.get_curr_life(obs)
            if curr_pos['z'] - dodger.start_pos['z'] > 10 or dodger.life == 0:
                success = False
                return total_reward, success
            
            if t < T:
                if running == False:
                    T = t + 1
                    break
                else:
                    dodger.agent_host.sendCommand(a)
                    time.sleep(dodger.sleep_time)
                    dodger.agent_host.sendCommand('move 0')

                    r, success = dodger.get_reward(obs, a)
                    total_reward += r
                    if success != None:
                        if success == True:
                            print("WIN!!!!!")
                        running = False
                        continue
                    s = dodger.get_curr_state(obs)
                    a = test_getAction(dodger, s, possible_actions)
                    print('s:', s, 'a:', a)
                len_epi = t + 1 - dodger.n
    return total_reward, success

# ========================================================
# main
# ========================================================
if __name__ == '__main__':
    # get agent_host
    agent_host = MalmoPython.AgentHost()
    world.handle_args(agent_host)

    # mission_xml_
    mission_xml_path =  world.update_mission_xml(mission_xml)
    print("mission_xml_path", mission_xml_path)

    # create ai
    dodger = Dodger(agent_host)

    # load the q_table trained before
    with open("q_table.pkl", 'rb') as f:
        dodger.q_table = pickle.load(f)
        # print(q_table)

    # record win rate
    num_wins = 0
    num_runs = 0
    win_rate = 0
    wr_table = [0] * 10
    wr_i = 0

    success_rate = [0] * 10
    num_reps = 1000

    moveable_blocks = list(range(-308, -318, -1))	# Agent自己可以走的格子
    possible_arrow_x_pos = list(range(455, 446, -1))	# 哪些格子上面可能会发射弩箭
    possible_arrow_x_pos.append(None)

    for rep in range(num_reps):
        os.system("clear")

        logging.info("win-rate in " + str(num_runs) + ": " + str(win_rate))
        print(wr_table)

        if dodger.life == 0 or rep == 0:
            agent_host.sendCommand("quit")
            world.start_mission(agent_host, mission_xml_path)
            time.sleep(1)
            world.refresh(agent_host, dodger)
            print("dodger.dispenser_pos", dodger.dispenser_pos)
            time.sleep(random.uniform(2, 3.5))
        logging.info("starting mission " + str(num_runs))

        total_reward, success = test_run(dodger)

        num_runs += 1
        num_wins += 1 if success == True else 0
        win_rate = round(num_wins / num_runs, 3)
        wr_table[wr_i] = win_rate
        if rep > 0 and rep % 100 == 0:
            wr_i += 1
            num_wins = num_runs = 0
            draw_success_rate(num_reps, wr_table)
            
        logging.info("end mission " + str(num_runs) + ": " + str(total_reward))
        world.soft_refresh(agent_host, dodger)
        time.sleep(random.uniform(0.1, 0.8))
        logging.info("reached max reps")
        print(wr_table)