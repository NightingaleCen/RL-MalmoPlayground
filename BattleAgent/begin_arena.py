from ddqn import DQNAgent
import sys
import time

try:
    from malmo import MalmoPython
except BaseException:
    import MalmoPython
import steve_agent
import json
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('config.ini')

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

with open('arena.xml', 'r') as file:
    missionXML = file.read()

my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10000))

state_size = int(config.get('DEFAULT', 'STATE_SIZE'))
action_size = int(config.get('DEFAULT', 'ACTION_SIZE'))
time_multiplier = 1
nn = DQNAgent(state_size, action_size)
done = False
batch_size = int(config.get('DEFAULT', 'BATCH_SIZE'))

# command line arguments
try:
    arg_check = sys.argv[1].lower()  # using arguments from command line
    if (arg_check not in ["zombie", "skeleton", "spider", "giant"]):
        print("\nInvalid mob type, defaulting to 1 zombie")
        mob_type = 'zombie'
        mob_number = 1
    else:
        mob_type = sys.argv[1]
        if (len(sys.argv) > 2):
            mob_number = int(sys.argv[2])
        else:
            mob_number = 1
        print(
            ("\nTESTING AGENT ON {} {}(S)").format(
                mob_number,
                mob_type.upper()))
except BaseException:
    print("\nError in argument parameters. Defaulting to 1 zombie")
    mob_type = 'zombie'
    mob_number = 1

nn_save = ""  # loading up previous save model if possible
if (len(sys.argv) > 3):
    try:
        nn_save = ("save/{}.h5").format(sys.argv[3])
        nn.load(nn_save)
        print("Save Model successfully imported")
    except BaseException:
        print("Save model not found!")
        sys.exit(1)
else:
    print("Incorrect arg format! Correct format: mob_type mob_number save_model")
    sys.exit(1)

time_start = time.time()
my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission(
            my_mission,
            my_client_pool,
            my_mission_record,
            0,
            "test")
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:", e)
            exit(1)
        else:
            time.sleep(1 / time_multiplier)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Starting Mission Error:", error.text)

# Disable natural healing
agent_host.sendCommand('chat /gamerule naturalRegeneration false')


time.sleep(1 / time_multiplier)
while len(world_state.observations) == 0:
    world_state = agent_host.getWorldState()
world_state_txt = world_state.observations[-1].text
world_state_json = json.loads(world_state_txt)
agent_name = world_state_json['Name']

agent_host.sendCommand(
    "chat /replaceitem entity " +
    agent_name +
    " slot.weapon.offhand minecraft:shield")

time.sleep(1 / time_multiplier)

print()
print("Mission running ", end=' ')


x = world_state_json['XPos']
y = world_state_json['YPos']
z = world_state_json['ZPos']
for i in range(mob_number):
    spawn_command = 'chat /summon {} {} {} {}'.format(
        mob_type, x - 8, y, z - 8 + (i * 2))
    if mob_type == 'zombie':
        spawn_command += ' {IsBaby:0}'
    agent_host.sendCommand(spawn_command)

time.sleep(1 / time_multiplier)

steve = steve_agent.Steve(mob_type)
# Loop until mission ends:

# keep track if we've seeded the initial state
have_initial_state = 0

mobs_left = mob_number
rewards = []
while world_state.is_mission_running:
    time.sleep(float(config.get('DEFAULT', 'TIME_STEP')) / time_multiplier)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("World State Error:", error.text)

    if world_state.number_of_observations_since_last_state > 0:

        msg = world_state.observations[-1].text
        ob = json.loads(msg)
        time_alive = int(time.time() - time_start)
        lock_on = steve.master_lock(ob, agent_host, is_testing=True)

        try:
            steve.get_state(ob, time_alive)
        except KeyError as k:
            print("Key Error Quit!", k)
            agent_host.sendCommand("quit")
            break

        # MAIN NN LOGIC
        # check if we've seeded initial state just for the first time
        if have_initial_state == 0:
            state = steve.get_state(ob, time_alive)
            have_initial_state = 1

        state = np.reshape(state, [1, state_size])
        action = nn.act(state)
        steve.perform_action(agent_host, action)  # send action to malmo
        msg = world_state.observations[-1].text
        ob = json.loads(msg)
        steve.get_mob_loc(ob)  # update entities in steve
        next_state = steve.get_state(ob, time_alive)
        lock_on = steve.master_lock(ob, agent_host, is_testing=True)

        if next_state[4] == 0:  # steve clearing arena
            arena_clear = True
        else:
            arena_bonus = False

        next_state = np.reshape(next_state, [1, state_size])

        state = next_state

        if (arena_bonus == True):  # just some quick spaghetti to get us out of NN loop after cleared arena hehe
            agent_host.sendCommand("quit")
            break


