# ==============================================================================
# MODULES
# ==============================================================================
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
import numpy

# ==============================================================================
# AI class
# ==============================================================================
class Dodger(object):
	def __init__(self, agent_host, alpha=0.4, gamma=.95, n=1):
		self.agent_host = agent_host	# init in main
		self.alpha = alpha				# learning rate
		self.gamma = gamma				# value decay rate
		self.n = n						# number of back steps to update
		self.epsilon = 0.1			# chance of taking a random action
		self.q_table = {}				
		self.start_pos = None			# init in world.refresh(...)
		self.dispenser_pos = None		# init in world.refresh(...)
		self.life = 0					# init in world.refresh(...)
		self.sleep_time	= 0.05			# time to sleep after action    0.05
		
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
		if len(A) == 0:
			A.append('move 0')
		
		# calculate q value based on the most recent state/action/reward
		# print("S", S)
		# print("A", A)
		# print("R", R)
		curr_s, curr_a, curr_r = S.popleft(), A.popleft(), R.popleft()
		TD_target = curr_r + self.gamma * numpy.max([self.q_table[S[-1]][a] for a in ['move 0', 'move 1']])
		TD_error = TD_target - self.q_table[curr_s][curr_a]
		self.q_table[curr_s][curr_a] += self.alpha * TD_error

	def get_reward(self, obs, prev_action):
		"""	get reward based on distance, life, action, and arrow avoidance
			args:	world observation	(use world.get_observations(...))
					prev_action			(use self.get_action(...))
			return:	reward value		(float)
					success flag		(True / False / None = still in progress)
		"""
		# reward = distance from start position * the following multipliers
		curr_pos = world.get_curr_pos(obs)
		dist = curr_pos['z'] - self.start_pos['z']
		cumul_multi = 1

		# initialize reward multipliers and success flag
		DAMAGE = -100
		COMPLETE = 100
		WAIT = 0.5
		AVOID = 3
		success = None
		
		# damaged: extremely low reward and success = False
		if curr_pos['x'] != self.start_pos['x']:
			success = False
			cumul_multi *= DAMAGE

		# complete: extremely high reward and success = True
		view_ahead = obs.get('view_ahead', [0])
		if view_ahead[0] == 'chest':
			success = True
			cumul_multi *= COMPLETE

		# waited: scale down reward
		if prev_action == 'move 0':
			cumul_multi *= WAIT
		
		# avoided arrow: scale up reward
		# 由于上面已经判断了是否被箭击中，那么判断avoid的条件只需要判断是否在可能被箭射中的位置
		dispenser_z = [dispenser[2] for dispenser in self.dispenser_pos]
		if int(curr_pos['z']) - 2 in dispenser_z:
			cumul_multi *= AVOID

		return dist * cumul_multi, success

	def get_action(self, curr_state, possible_actions):
		"""	get best action using epsilon greedy policy
			args:	current state		(use self.get_curr_state(obs))
					possible actions	(["move 1", "move 0"])
			return:	action				("move 1" or "move 0")
		"""
		# new state
		# NOTE: maybe make it so agent always moves in a new state?
		if curr_state not in self.q_table:
			self.q_table[curr_state] = {}
			for a in possible_actions:
				self.q_table[curr_state][a] = 0
		
		# chance to choose a random action
		# NOTE: maybe random chance to move instead?
		if random.random() < self.epsilon:
			action_i = random.randint(0, len(possible_actions) - 1)
		else:
			values = [self.q_table[curr_state][a] for a in possible_actions]
			action_i = numpy.argmax(values)
		
		return possible_actions[action_i]
		# get the best action based on the greatest q-val(s)
		

	def get_curr_state(self, obs):
		"""	get a simplified, integer-based version of the environment
			args:	world observations	(use world.get_observations(...))
			return:	state 				((curr z, arrow₁ x, arrow₂ x, ...))
		"""
		state = []

		# get current z-position rounded down
		curr_pos = world.get_curr_pos(obs)
		curr_z = int(curr_pos['z']) - 1
		state.append(curr_z)

		# get arrow x-positions, ordered by increasing z-positions
		curr_arrow_pos = world.get_arrow_pos(obs)
		for x, y, z in self.dispenser_pos:	# 从这个地方是获取的从小到大的z坐标排序好的dispenser
			if int(z) in curr_arrow_pos:	# 因为curr_arrow_pos是一个无序的字典，
				# 如需要判断，只需要用curr_arrow_pos中取出即可
				state.append(curr_arrow_pos[int(z)])
			else:
				state.append(None)

		# (curr_pos[z], arrow_pos[z₁] = x₁, arrow_pos[z₂] = x₂, ...)
		return tuple(state)
		


	def run(self):
		"""	observations → state → act → reward ↩, and update q table
			return:	total reward		(cumulative int reward value of the run)
					success flag		(True / False)
		"""
		# history of states/actions/rewards
		S, A, R = deque(), deque(), deque()
		
		# either you move or you don't
		possible_actions = ['move 1', 'move 0']
		
		# returns total reward and success flag
		total_reward, success = 0, None

		# initialize terminating state
		terminate_s = 'ENDDING'
		self.q_table[terminate_s] = {}
		for a in possible_actions:
			self.q_table[terminate_s][a] = 0

		# run until damaged
		running = True
		while running:

			# get initial state/action/reward
			obs = world.get_observations(self.agent_host)
			s0 = self.get_curr_state(obs)
			a0 = self.get_action(s0, possible_actions)
			r0 = 0
			S.append(s0)
			A.append(a0)
			R.append(r0)

			# continuously get observations
			T = sys.maxsize		# T就是预设，大概什么时候是episode的终点，初始化给定一个很大的值
			for t in range(sys.maxsize):	# t 就是在一个episode中，采样到第几个状态了
				obs = world.get_observations(self.agent_host)
				# death or out of bounds glitching ends the run
				curr_pos = world.get_curr_pos(obs)
				self.life = world.get_curr_life(obs)
				if curr_pos['z'] - self.start_pos['z'] > 10 or self.life == 0:
					# episode finish: end state and get final reward
					success = False
					return total_reward, success

				if t < T:
					# 在每一轮开始之前，已经知道了上一轮状态、上一轮的action、上一轮action后的reward
					# 现在需要进行的是1.根据上次行动的结果reward和running_flag来判断是否结束
					# 2.执行上一轮根据当前状态得到的action
					# 3.执行action后得到一个reward、running_flag
					# 4.执行action后观察状态、选择动作
					# episode running: act and get state/action/reward
					if running == False:
						# 说明当前episode已经采样结束了
						T = t + 1	# 当前episode的最大采样步数T就是t+1
						S.append(terminate_s)	# 说明当前的状态就是S_T+1，也就是terminate_s
					
					else:
						# act (move or wait)
						self.agent_host.sendCommand(A[-1])
						time.sleep(self.sleep_time)
						self.agent_host.sendCommand('move 0')
						
						# get reward and check if episode is finished 
						r, success = self.get_reward(obs, A[-1])
						R.append(r)
						total_reward += r
						if success != None:
							if success == True:
								print("WIN!!!!!!")
							running = False
							continue	# 到了终点或者死掉了就不再加入当前状态，而是加入end状态

						# get state/action
						s = self.get_curr_state(obs)
						a = self.get_action(s, possible_actions)
						S.append(s)
						A.append(a)
						print('s:',s, 'a:', a)
				# end of episode: update q table
				len_epi = t + 1 - self.n
				if S[0] == terminate_s:	# 排除掉只剩下一个S的情况
					break
				if len_epi >= 0:
					self.update_q_table(len_epi, S, A, R, T)

		return total_reward, success

	def hard_coded_run(self, wait_block, arrow_x_pos):
		"""	guarantee move when agent on wait_block and arrow on arrow_x_pos
			return:	success flag		(True / False)
		"""	
	

			# get initial state/action/reward
			
			# death or out of bounds glitching ends the run
			
			# act
			
			# win/lose condition

