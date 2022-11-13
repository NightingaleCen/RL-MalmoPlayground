# ==============================================================================
# MODULES
# ==============================================================================
import ppo
import torch
import numpy as np
try:
    from malmo import MalmoPython
except:
    import MalmoPython				# Malmo
import logging						# world debug prints
import time							# sleep for a few ticks every trial
import random						# random chance to choose actions
import world						# world observation
import sys							# max int tau
from collections import deque		# states/actions/rewards history


R_DAMAGED = -10
R_COMPLETE = 10
R_AVOID_ARW = 3		# avoid the arrow, so scale up the reward
R_WAITING = 0.9		# wait but wasting time, so scale down the reward
# ==============================================================================
# AI class
# ==============================================================================
class Dodger(object):
	def __init__(self, agent_host, num_arrow, alpha=0.001, gamma=.95, n=1):
		self.agent_host = agent_host	# init in main
		self.alpha = alpha				# learning rate
		self.gamma = gamma				# value decay rate
		self.n = n						# number of back steps to update
		self.epsilon = 0.2			# chance of taking a random action
		self.q_table = {}				
		self.start_pos = None			# init in world.refresh(...)
		self.dispenser_pos = None		# init in world.refresh(...)
		self.life = 0					# init in world.refresh(...)
		self.sleep_time	= 0.05			# time to sleep after action NOTE:0.05
		self.num_state = 1
		self.net = None
		self.start_z = -316.5
		# self.start_x = self.curr_x= 415.5
	# USE FOR 1 ARROW TESTING PURPOSES ONLY
	def print_1arrow_q_table(self, moveable_blocks, possible_arrow_x_pos):
		"""	prints a formatted q-table for an 1 arrow run
			args:	moveable blocks		(blocks the agent can walk on (rows))
					arrow x positions	(possible arrow x positions (columns))
		"""
		

	# USE FOR 1 ARROW HARD CODED RUN TESTING PURPOSES ONLY 
	def print_hc_wr_table(self, wait_block, possible_arrow_x_pos, wr_table):
		"""	prints a formatted win-rate table for a hard coded 1 arrow run
			args:	wait block			(block the agent can walk on (rows))
					arrow x positions	(possible arrow x positions (columns))
					win-rate table		(win-rates per possible arrow x pos)
		"""
		# print arrow x positions (x-axis)
		
		
		# print wait block (y-axis)
		
		
		# print win-rate for each arrow x position
		
		
	def update_q_table(self, tau, S, A, R, T):
		"""	performs relevant updates for state tau
		
			args:	tau				(integer state index to update)
					states deque	
					actions deque	
					rewards deque	
					term state index
		"""
		# upon terminating state, A is empty
		
		
		# calculate q value based on the most recent state/action/reward
		

	def get_reward(self, obs, prev_action):
		"""	get reward based on distance, life, action, and arrow avoidance
			args:	world observation	(use world.get_observations(...))
					prev_action			(use self.get_action(...))
			return:	reward value		(float)
					success flag		(True / False / None = still in progress)
		"""
		# reward = distance from start position * the following multipliers
		 
		
		# initialize reward multipliers and success flag
		
		# damaged: extremely low reward and success = False
		

		# complete: extremely high reward and success = True

		# waited: scale down reward
		
		
		# avoided arrow: scale up reward
		

	# def state2tensor(self, curr_state):
	# 	''' state :tuple(.., [.., ..])->tensor that canbe sent into net		
	# 	'''
	# 	state = [curr_state[0]]
	# 	for item in curr_state[1]:
	# 		state.append(item)
	# 	return torch.tensor(state)

	def get_action(self, possible_actions, policy):
		"""	get best action using epsilon greedy policy
			args:	current state		(use self.get_curr_state(obs))
					possible actions	(["move 1", "move 0"])
			return:	action				("move 1" or "move 0")
		"""
		# NOTE
		# new state
		# NOTE: maybe make it so agent always moves in a new state?
		if random.random() < self.epsilon:
			action = possible_actions[random.randint(0, 1)]
		else:
			p = policy.detach().numpy().tolist()
			p[1] = 1.0 - p[0]
			# p.append(float(policy[0]))
			# p.append(float(policy[1]))
			# print(policy.squeeze().detach().numpy().tolist())
			action_num = int(np.random.choice(2, 1, p=p))
			action = possible_actions[action_num]
		# chance to choose a random action
		# NOTE: maybe random chance to move instead?
		return action
			
		# get the best action based on the greatest q-val(s)
		

	def get_curr_state(self, obs):
		"""	get a simplified, integer-based version of the environment
			args:	world observations	(use world.get_observations(...))
			return:	state 				((curr z, arrow₁ x, arrow₂ x, ...))
		"""
		# get current z-position rounded down
		agent_pos = world.get_curr_pos(obs)
		print(float(agent_pos['z']))
		agent_cur_z = float(agent_pos['z']) - self.start_z 	# 从起始点算起
		self.curr_x = float(agent_pos['x'])
		# dispenser_pos = world.get_dispenser_pos(self.agent_host, obs, self.start_pos)
		# print("dispenser_pos", dispenser_pos)

		# get arrow x-positions, ordered by increasing z-positions
		arrow_dic = world.get_arrow_pos(obs)
		# print('arrow_dic', arrow_dic)
		arrow_dic_keys = []
		# get agent_life
		agent_life = world.get_curr_life(obs)
		# arrow_dic_keys = [int(key) for key in arrow_dic.keys()].sort()
		for key in arrow_dic.keys():
			# print('key:',int(key))
			arrow_dic_keys.append(int(key))
			arrow_dic[int(key)] -= 451.5
		arrow_x_positions = []
		print('arrow_dic_keys:',arrow_dic_keys)
		# arrow_dic_keys.sort()
		# print('now agent"s z:', )
		state = [0.0]*4
		state[0] = agent_cur_z
		# if len(arrow_dic_keys) != 0:
		# 	i = 1
		# 	for z in arrow_dic_keys:
		# 		# state.append(float(arrow_dic[z]))
		# 		state.append()
		# 		# state.append()
		# 		break
		# (curr_pos[z], arrow_pos[z₁] = x₁, arrow_pos[z₂] = x₂, ...)
		# print('arrow_x_positions:', arrow_x_positions)
		# if len(arrow_dic_keys) != 0:
			# state.append(float(arrow_x_positions[0]))
		# for arrow_pos in arrow_x_positions:
		# 	state.append(float(arrow_pos))
		for (index, item) in enumerate(self.dispenser_pos):
			# print("item", type(index))
			# print("int(item[int(index)])", item[])
			if int(item[2]) in arrow_dic_keys:
				state[index+1] = 1.0
		return state, agent_life
	
	# def is_damaged(self, state, next_state, arrow_dic, pre_life):
	def is_damaged(self, pre_life):
		# for key in arrow_dic.keys():
		# 	if (state[0] + self.start_z <= key and key <= next_state[0] + self.start_z \
		# 		and 450 <= arrow_dic[key] + 446 and arrow_dic[key] + 446 <= 453):	# 之前是451-452
		# 		return True
		# return False
		return self.life < pre_life
		# return self.curr_x != self.start_x

	def is_complete(self, next_state):
		return next_state[0] >= 8.0

	def get_distance(self, next_state):
		return abs(next_state[0])

	def step(self, next_state, action, pre_life):
		r_mult = 1
		success = False
		done = False
		if action == 1:	# 向前进
			if self.is_damaged(pre_life):
				r_mult *= R_DAMAGED
				success = False
				done = True
			elif self.is_complete(next_state):
				r_mult *= R_COMPLETE
				success = True
				done = True
			else:
				r_mult *= R_AVOID_ARW	
				success = done = False

		else:		# waiting
			if self.is_damaged(pre_life):
				r_mult *= R_DAMAGED
				success = False
				done = True
			elif self.is_complete(next_state):
				r_mult *= R_COMPLETE
				success = True
				done = True
			else:
				r_mult *= R_WAITING
				success = done = False
		reward = r_mult * self.get_distance(next_state)
		return reward, done, success



	def run(self):
		"""	observations → state → act → reward ↩, and update q table
			return:	total reward		(cumulative int reward value of the run)
					success flag		(True / False)
		"""

		self.num_state = 1 + len(self.dispenser_pos)
		self.net = ppo.PPO(self.num_state, 2, 1, self)
		MAX_STEP = 500
		done = False	# done == True iff:被箭给射死(life==0) or 到达了终点
		total_reward = 0
		success = False
		observations = world.get_observations(self.agent_host)	# 从环境中获取到observations(字典)
		state, self.life = self.get_curr_state(observations)	# 从字典observations里面提取到需要的整数状态信息，箭字典
		pre_life = self.life
		while not done:
			for t in range(MAX_STEP):
				print('t------', t)
				# print('torch.tensor(state).float()', torch.tensor(state).float().reshape(1, -1))
				policy = self.net.pi(torch.tensor(state).float().reshape(1, -1), softmax_dim=1).squeeze()
				# print('policy is', policy)
				a = self.get_action(["move 0", "move 1"], policy)	# "move 1" or "move 0"
				# if '1' in a:
				self.agent_host.sendCommand(a)	# 控制agent移动
				time.sleep(self.sleep_time)	# 间隔一定的时间
					# self.agent_host.sendCommand('move 0')
				a = 1 if '1' in a else 0		# 1 or 0
				observations = world.get_observations(self.agent_host)	# 从环境中获取到observations(字典)
				next_state, self.life = self.get_curr_state(observations)	# 得到下一个状态
				r, done, success = self.step(next_state, a, pre_life)	# 获得奖励、完成情况
				total_reward += r
				if self.life == 0:
					done = True
					success = False
				# 另外实现一个判断damaged的办法：比较上次和现在的life值，如果减小则说明damaged
				print('state::::', state)
				print('a:::', a)
				print('r:::', r)
				print('next_state', next_state)
				print()
				print()
				# print('policy[a]', policy[a])
				if done:
					if success:
						print('Yes, success!')
					else:
						print('No, you are damaged!')
				data = [state, a, r, next_state, policy[a], done]
				# print('data', t, 'is~~~~', data)
				self.net.put_data(data)
				# TODO:更新环境，but how to update the arrows?
				state, pre_life = next_state, self.life
				if done:
					self.agent_host.sendCommand('move 0')
					break
			if t >= 3:
				self.net.train_net()
		
		return total_reward, success
		

		# history of states/actions/rewards

		# either you move or you don't
		
		# returns total reward and success flag
		
		# initialize terminating state 

		# run until damaged


			# get initial state/action/reward


			# continuously get observations
	
				# death or out of bounds glitching ends the run
				
					# episode finish: end state and get final reward
					
					
					# episode running: act and get state/action/reward
					
						# act (move or wait)
						
						
						# get reward and check if episode is finished 
						
						
						# get state/action
		
				# end of episode: update q table
		# return None, None			

	def hard_coded_run(self, wait_block, arrow_x_pos):
		"""	guarantee move when agent on wait_block and arrow on arrow_x_pos
			return:	success flag		(True / False)
		"""	
	

			# get initial state/action/reward
			
			# death or out of bounds glitching ends the run
			
			# act
			
			# win/lose condition

