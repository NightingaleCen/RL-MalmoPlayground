from malmo import MalmoPython
import os
import sys
import time
import random
from random import randrange as rand
from collections import deque
from tetris_game import *
import pickle
import copy
import numpy
from numpy import zeros

# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # flush print output immediately

rewards_map = {'inc_height': -20, 'clear_line': 50, 'holes': -20, 'top_height':-100}

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Tetris!</Summary>
        </About>
        <ServerSection>
                    <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                <DrawingDecorator>
                    <DrawLine x1="2" y1="56" z1="22" x2="2" y2="72" z2="22" type="obsidian"/>
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Creative">
            <Name>MalmoTutorialBot</Name>
            <AgentStart>
                <Placement x="2.5" y="73" z="22.8" yaw="180"/>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats/>
                <ContinuousMovementCommands turnSpeedDegs="180"/>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''

def magic(X):
    return ''.join(str(i) for i in X)

QLfilename = 'maltrisQLTable.save'
QLgraphfilename = 'maltrisQLGraphData.save'

class TetrisAI:
    def __init__(self, game, alpha=0.3, gamma=1, n=1):
        self.epsilon = 0.3
        self.q_table = {}
        self.n, self.alpha, self.gamma = n, alpha, gamma
        self.gamesplayed = 0
        self.listGameLvl = []
        self.listClears = []
        self.game = game

    def saveQtable(self):
        f = open(QLfilename, 'wb')
        pickle.dump(self.gamesplayed, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        f2 = open(QLgraphfilename, 'wb')
        pickle.dump(self.gamesplayed, f2, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.listGameLvl, f2, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.listClears, f2, protocol=pickle.HIGHEST_PROTOCOL)
        f2.close()

    def loadQtable(self):
        f = open(QLfilename, 'rb')
        self.gamesplayed = pickle.load(f,encoding='iso-8859-1')
        self.q_table = pickle.load(f,encoding='iso-8859-1')
        f.close()
        f2 = open(QLgraphfilename, 'rb')
        self.gamesplayed = pickle.load(f2,encoding='iso-8859-1')
        self.listGameLvl = pickle.load(f2,encoding='iso-8859-1')
        self.listClears = pickle.load(f2,encoding='iso-8859-1')
        f2.close()

    def run(self, agent_host):
        states, actions, rewards = deque(), deque(), deque()
        curr_reward = 0
        done_update = False
        game_overs = 0
        self.loadQtable() #uncomment to load Q-table values
        while not done_update:
            init_state = self.get_curr_state()
            possible_actions = self.get_possible_actions()
            next_action = self.choose_action(init_state, possible_actions)
            # print("init_state", init_state)
            states.append(init_state)
            actions.append(self.normalize(self.pred_insta_drop2(next_action)))
            # print("actions", actions)
            rewards.append(0)

            T = sys.maxsize
            for t in range(sys.maxsize):
                # print("episode{}".format(str(t)))
                time.sleep(0.1)
                if t < T:
                    curr_reward = self.act(next_action)
                    rewards.append(curr_reward)
                    # curr_state = copy.deepcopy(self.get_curr_state())
                    # # print("now_state", curr_state)
                    # states.append(curr_state)
                    # next_action = self.choose_action(curr_state, self.get_possible_actions())
                    # actions.append(next_action)
                    # time.sleep(0.5)

                    if self.game.gameover == True:
                        self.listGameLvl.append(self.game.level)
                        self.listClears.append(self.game.line_clears)
                        print("level {}".format(self.game.level))
                        print("line clears {}".format(self.game.line_clears))
                        self.game.start_game()
                        game_overs += 1
                        self.gamesplayed += 1

                        if game_overs == 5:
                            print("Best attempt, gamesplayed: ", self.gamesplayed,
                                  " avg levels: ", numpy.mean(self.listGameLvl),
                                  " avg clears: ", numpy.mean(self.listClears))
                            game_overs = 0

                    ########################################
                    ######TODO: fill your code here#########
                    ########################################
                    curr_state = copy.deepcopy(self.get_curr_state())
                    # print("now_state", curr_state)
                    states.append(curr_state)
                    next_action = self.choose_action(curr_state, self.get_possible_actions())
                    actions.append(self.normalize(self.pred_insta_drop2(next_action)))
                    time.sleep(0.5)

                tau = t - self.n + 1
                if tau >= 0:
                    self.update_q_table(tau, states, actions, rewards, T)

                if tau == T - 1:
                    while len(states) > 1:
                        tau = tau + 1
                        self.update_q_table(tau, states, actions, rewards, T)
                    done_update = True
                    break

                if t%1000 == 0:
                    self.saveQtable()  # uncomment to save Q-table values
                    print("------------------Saving Qtable------------")
                    time.sleep(0.1)

    def act(self, action):
        # 本函数为与环境交互的函数，action为2维向量，第一维表示平移多少，第二维表示旋转多少
        for i in range(action[1]):
            self.game.rotate_piece()
        # 旋转好了之后，一路下落
        self.game.move(action[0] - self.game.piece_x)
        m_score = self.score(self.pred_insta_drop2(action))
        # self.game.insta_drop_no_draw()
        self.game.insta_drop()
        return m_score

    def get_curr_state(self):
        # 从倒数第二行开始，倒序取每行的信息
        board = self.game.board[-2::-1]
        # 倒序开始枚举，枚举到一行全0，取其下面两行的块信息，作为state
        for i, row in enumerate(board):
            if all(j == 0 for j in row):
                if i < 2:
                    new_state = board[0:2]
                    # 0/1二值化操作
                    new_state = [[1 if x!= 0 else x for x in row]for row in new_state]
                    return new_state
                else:
                    new_state = board[i-2:i]
                    new_state = [[1 if x!= 0 else x for x in row]for row in new_state]
                    return new_state

    def normalize(self, state):
        # 把整个游戏的game board变成了输入的state
        board = state[-2::-1]
        for i, row in enumerate(board):
            if all(j == 0 for j in row):
                if i < 2:
                    new_state = board[0:2]
                    new_state = [[1 if x!= 0 else x for x in row]for row in new_state]
                    return new_state
                else:
                    new_state = board[i-2:i]
                    new_state = [[1 if x!= 0 else x for x in row]for row in new_state]
                    return new_state

    def get_possible_actions(self):
        actions = []
        action = (0,0)
        ########################################
        ######TODO: fill your code here#########
        ########################################
        for i in range(4):
            piece_x = 0
            piece_y = self.game.piece_y

            while piece_x <= self.game.rlim - len(self.game.piece[0]):
                if not check_collision(self.game.board,
                                       self.game.piece,
                                       (piece_x, piece_y)):
                    if action not in actions:
                        actions.append(action)
                piece_x += 1
                action = (action[0] + 1, action[1])
                piece_y = self.game.piece_y
            self.game.rotate_piece()
            action = (0, action[1] + 1)
        self.game.rotate_piece()

        return actions

    def score(self, board):
        current_r = 0
        complete_lines = 0
        highest = 0

        ########################################
        ######TODO: fill your code here#########
        ########################################
        new_board = board[-2::-1]
        for row in new_board:
            if 0 not in row:
                complete_lines += 1
        current_r += complete_lines * rewards_map['clear_line']

        for i, row in enumerate(new_board):
            if all(j == 0 for j in row):
                highest = i

        current_r += (highest - 2) * rewards_map['top_height']

        heights = zeros((len(new_board[0]),), dtype=int)
        for i, row in enumerate(new_board):
            for j, col in enumerate(row):
                if col != 0:
                    heights[j] = i + 1
        aggregate_height = sum(heights)
        current_r += aggregate_height * rewards_map['inc_height']

        return current_r

    def rotate_piece(self, piece, piece_x, piece_y, board):
        new_piece = rotate_clockwise(piece)
        if not check_collision(board, new_piece, (piece_x, piece_y)):
            return new_piece
        else:
            return piece

    def pred_insta_drop(self, piece, piece_x, piece_y):
        # 相比drop2，没有action的空间
        new_board = copy.deepcopy(self.game.board)

        while not check_collision(new_board,
                           piece,
                           (piece_x, piece_y+1)):
            piece_y += 1

        piece_y += 1
        new_board = join_matrixes(
            new_board,
            piece,
            (piece_x, piece_y))

        return new_board

    def pred_insta_drop2(self, action):
        # 深拷贝一个新游戏板，用于预测下一个state长啥样
        new_board = copy.deepcopy(self.game.board)
        new_piece = self.game.piece
        new_piece_x = self.game.piece_x
        new_piece_y = self.game.piece_y

        # action第二个维度，表示要将piece旋转几下
        for i in range(action[1]):
            new_piece = self.rotate_piece(new_piece, new_piece_x, new_piece_y, new_board)

        # action第一个维度，表示要将piece平移到什么位置
        new_piece_x = action[0] - new_piece_x + 1
        # 防止溢出
        if new_piece_x < 0:
            new_piece_x = 0
        if new_piece_x > cols - len(new_piece[0]):
            new_piece_x = cols - len(new_piece[0])

        # piece下落
        while not check_collision(new_board,
                           new_piece,
                           (new_piece_x, new_piece_y+1)):
            new_piece_y += 1

        new_piece_y += 1
        # 下落完毕，整合进入board中
        new_board = join_matrixes(
            new_board,
            new_piece,
            (new_piece_x, new_piece_y))

        return new_board

    def choose_action(self, state, possible_actions):
        curr_state = magic(state)
        # print("current state", curr_state)
        #######################################
        ######TODO: fill your code here#########
        ########################################
        # 没出现过的新state，加入q_table
        if curr_state not in self.q_table:
            self.q_table[curr_state] = {}
        # 逐个查找其他state
        for action in possible_actions:
            state = magic(self.normalize(self.pred_insta_drop2(action)))
            if state not in self.q_table[curr_state]:
                self.q_table[curr_state][state] = 0

        e = numpy.random.random()
        if e < self.epsilon:
            action = numpy.random.randint(0, len(possible_actions) - 1)
            best_action = possible_actions[action]
        else:
            best_action = possible_actions[0]
            best_next_state = magic(self.normalize(
                self.pred_insta_drop2(possible_actions[0])))
            qvals = self.q_table[curr_state]
            for action in possible_actions:
                state = magic(self.normalize(self.pred_insta_drop2(action)))
                if qvals[state] >= qvals[best_next_state]:
                    best_next_state = state
                    best_action = action
        return best_action

    def update_q_table(self, tau, S, A, R, T):
        curr_s, curr_a, curr_r = magic(S.popleft()), A.popleft(), R.popleft()
        ########################################
        ######TODO: fill your code here#########
        ########################################
        state = magic(curr_a)
        G = sum([self.gamma ** i * R[i] for i in range(len(S))])
        if tau + self.n < T:
            G += self.gamma ** self.n * self.q_table[magic(S[-1])][magic(A[-1])]

        print("episode reward {}".format(G))
        old_q = self.q_table[curr_s][state]
        # 使用baseline方法，减小variance
        self.q_table[curr_s][state] = old_q + self.alpha * (G - old_q)

if __name__ == "__main__":
    random.seed(0)

    #Initialize agent_host
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print("ERROR:",e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    #Initialize Mission
    mission = MalmoPython.MissionSpec(missionXML, True)
    mission.allowAllChatCommands()
    mission.forceWorldReset()
    mission_record = MalmoPython.MissionRecordSpec()

    #Build Tetris Board
    left_x, right_x = -1, -1+cols+1
    bottom_y, top_y = 68, 68+rows+1
    z_pos = 3
    mission.drawLine( left_x, bottom_y, z_pos, left_x, top_y, z_pos, "obsidian" )
    mission.drawLine( right_x, bottom_y, z_pos, right_x, top_y, z_pos, "obsidian" )
    mission.drawLine( left_x, bottom_y, z_pos, right_x, bottom_y, z_pos, "obsidian" )
    mission.drawLine( left_x, top_y, z_pos, right_x, top_y, z_pos, "obsidian" )
    for i in range(-1,cols):
        mission.drawLine(i, bottom_y, z_pos-1, i, bottom_y+rows, z_pos-1, "quartz_block")

    #Attempt to start Mission
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( mission, mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    #Loop until mission starts
    print("Waiting for the mission to start")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()
    print("Mission running")

    numIter = 1
    n = 1
    my_game = TetrisGame(agent_host)
    my_AI = TetrisAI(my_game)
    print("n=", n)
    for n in range(numIter):
        my_AI.run(agent_host)
    print(my_AI.q_table)
