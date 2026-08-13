import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
class Agent():
    #玩家先后手，随机行动的概率，学习率
    def __init__(self,name,epsilon, learning_rate):
        self.name = name
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.value=np.zeros((3,3,3,3,3,3,3,3,3))
        self.outcome=np.zeros(9).astype(np.int8)
    def reset(self):
        self.outcome = np.zeros(9).astype(np.int8)
    def act(self,state):
        state_1=state.copy()
        empty = np.where(state_1 == 0)[0]
        if np.random.rand()<self.epsilon:
            state_1[np.random.choice(empty)]=self.name
        else:
            temp_value=np.zeros(len(empty))
            for i in range(len(empty)):
                temp_state=state_1.copy()
                temp_state[empty[i]]=self.name
                temp_value[i]=self.value[tuple(temp_state)]
            choose=np.argmax(temp_value)
            state_1[empty[choose]]=self.name
        error=self.value[tuple(state_1)]-self.value[tuple(self.outcome)]
        self.value[tuple(self.outcome)]+=error*self.learning_rate
        self.outcome=state_1.copy()
        return state_1
#输赢判断函数
def isWin(state,name):
    T=np.repeat(name,3)
    winner=0
    if 0 not in state:
        winner=3
    if (state[0:3]==T).all() or (state[3:6]==T).all() or (state[6:9]==T).all():
        winner=name
    if (state[0:7:3]==T).all() or (state[1:8:3]==T).all() or (state[2:9:3]==T).all():
        winner=name
    if (state[0:9:4]==T).all() or (state[2:7:2]==T).all():
        winner=name
    return winner

Agent1=Agent(name=1,epsilon=0.1,learning_rate=0.1)
Agent2=Agent(name=2,epsilon=0.1,learning_rate=0.1)
Trial=30000
Winner=np.zeros(Trial)
for i in range(Trial):
    if i==20000:
        Agent1.epsilon=0
        Agent2.epsilon=0
    Agent1.reset()
    Agent2.reset()
    winner=0
    state=np.zeros(9).astype(np.int8)
    while winner==0:
        outcome=Agent1.act(state)
        winner=isWin(outcome,Agent1.name)
        if winner==1:
            Agent1.value[tuple(outcome)]=1
            Agent2.value[tuple(state)]=-1
        elif winner==0:
            state=Agent2.act(outcome)
            winner = isWin(state, Agent2.name)
            if winner==2:
                Agent2.value[tuple(state)] = 1
                Agent1.value[tuple(outcome)] = -1
            elif winner==3:
                Agent2.value[tuple(state)] = 0
                Agent1.value[tuple(outcome)] = 0
    Winner[i]=winner

# ========== 数据可视化 ==========
window = 500
agent1_win = (Winner == 1).astype(float)
agent2_win = (Winner == 2).astype(float)
draw = (Winner == 3).astype(float)

def moving_avg(data, w):
    return np.convolve(data, np.ones(w)/w, mode='valid')

x_smooth = np.arange(window-1, Trial)
plt.figure(figsize=(12, 5))
plt.plot(x_smooth, moving_avg(agent1_win, window), label='Agent1 胜率', linewidth=1)
plt.plot(x_smooth, moving_avg(agent2_win, window), label='Agent2 胜率', linewidth=1)
plt.plot(x_smooth, moving_avg(draw, window), label='平局率', linewidth=1)
plt.axvline(20000, color='gray', linestyle='--', alpha=0.5, label='epsilon=0')
plt.title(f'滑动平均胜率 (窗口={window}局)')
plt.xlabel('局数')
plt.ylabel('比率')
plt.legend()
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('对弈结果可视化.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n可视化完成！")
