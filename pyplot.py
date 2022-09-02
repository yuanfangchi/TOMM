"""
===============
Pyplot tutorial
===============

An introduction to the pyplot interface.
"""

import matplotlib.pyplot as plt
# t = [100, 200, 300]
#
# fig, ax = plt.subplots()
# ax.plot(t, [-0.03057, -0.03796, -0.04113], 'r-o', label='Agent A Batch Loss')
# ax.plot(t, [-0.1359, -0.1666, -0.1822], 'b-s', label='Agent B Batch Loss')
# ax.plot(t, [-0.11115, -0.2174, -0.26935], 'g--^', label='Agent C Batch Loss')
#
# legend = ax.legend(loc='lower left', shadow=True, fontsize='small')
# plt.ylabel('loss')
# plt.xlabel('# of batches')
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(t, [-0.03394, -0.03763, -0.04258], 'r-o', label='Agent A Batch Loss')
# ax.plot(t, [-0.1357, -0.1608, -0.1769], 'b-s', label='Agent B Batch Loss')
# ax.plot(t, [-0.1623, -0.2129, -0.2414], 'g--^', label='Agent C Batch Loss')
#
# legend = ax.legend(loc='upper right', shadow=True, fontsize='small')
# plt.ylabel('loss')
# plt.xlabel('# of batches')
# plt.show()

#t = [100, 200, 300, 400, 500]

# fig, ax = plt.subplots()
# ax.plot(t, [-0.082743, -0.098729, -0.11468, -0.1419, -0.15441], 'r-o', label='Agent A Batch Loss')
# ax.plot(t, [-0.08719, -0.10470, -0.11515, -0.1336, -0.1498], 'b-s', label='Agent B Batch Loss')
# ax.plot(t, [-0.08496, -0.09969, -0.12015, -0.1424, -0.17902], 'g--^', label='Agent C Batch Loss')
#
# legend = ax.legend(loc='lower left', shadow=True, fontsize='small')
# plt.ylabel('loss')
# plt.xlabel('# of batches')
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(t, [-0.0853, -0.09618, -0.1191, -0.13993, -0.17708], 'r-o', label='Agent A Batch Loss')
# ax.plot(t, [-0.17494, -0.2247, -0.2392, -0.2452, -0.27421], 'b-s', label='Agent B Batch Loss')
# ax.plot(t, [-0.2333, -0.2931, -0.30510, -0.2921, -0.29146], 'g--^', label='Agent C Batch Loss')
#
# legend = ax.legend(loc='upper right', shadow=True, fontsize='small')
# plt.ylabel('loss')
# plt.xlabel('# of batches')
# plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# labels = ['Agent A', 'Agent B', 'Agent C']
# hit_1 = [0.035, 0.104, 0.140]
# hit_3 = [0.075, 0.174, 0.216]
# hit_5 = [0.100, 0.212, 0.254]
# hit_10 = [0.128, 0.241, 0.284]
# hit_20 = [0.128, 0.241, 0.284]
# mrr = [0.062, 0.148, 0.187]

# labels = ['Agent B', 'Agent A', 'Agent C']
# hit_1 = [0.061, 0.104, 0.143]
# hit_3 = [0.118, 0.172, 0.219]
# hit_5 = [0.150, 0.208, 0.261]
# hit_10 = [0.179, 0.234, 0.291]
# hit_20 = [0.179, 0.234, 0.291]
# mrr = [0.097, 0.146, 0.190]

# labels = ['Agent C', 'Agent A', 'Agent B']
# hit_1 = [0.062, 0.113, 0.133]
# hit_3 = [0.129, 0.186, 0.217]
# hit_5 = [0.167, 0.222, 0.254]
# hit_10 = [0.199, 0.248,0.284]
# hit_20 = [0.199, 0.248,0.284]
# mrr = [0.104, 0.157, 0.183]

# labels = ['Agent A', 'Agent B', 'Agent C']
# hit_1 = [0.283, 0.353, 0.362]
# hit_3 = [0.339, 0.385, 0.415]
# hit_5 = [0.369, 0.417, 0.442]
# hit_10 = [0.380, 0.433, 0.461]
# hit_20 = [0.380, 0.433, 0.461]
# mrr = [0.317, 0.376, 0.394]

# labels = ['Agent B', 'Agent A', 'Agent C']
# hit_1 = [0.199, 0.280, 0.311]
# hit_3 = [0.345, 0.370, 0.404]
# hit_5 = [0.381, 0.399, 0.437]
# hit_10 = [0.390, 0.417, 0.456]
# hit_20 = [0.390, 0.417, 0.456]
# mrr = [0.271, 0.329, 0.364]
#
labels = ['Agent C', 'Agent A', 'Agent B']
hit_1 = [0.312, 0.145, 0.316]
hit_3 = [0.368, 0.372, 0.395]
hit_5 = [0.400, 0.417, 0.443]
hit_10 = [0.411, 0.428, 0.457]
hit_20 = [0.411, 0.428, 0.457]
mrr = [0.346, 0.261, 0.365]

x = np.arange(len(labels))  # the label locations
width = 0.125  # the width of the bars

fig, ax = plt.subplots()

#plt.ylim((0, 0.6))
hit_1 = ax.bar(x - 3 * width, hit_1, width, label='Hit@1', color='r', hatch='/')
hit_3 = ax.bar(x - 2 * width, hit_3, width, label='Hit@3', color='y', hatch='-')
hit_5 = ax.bar(x - width, hit_5, width, label='Hit@5', color='g', hatch='x')
hit_10 = ax.bar(x, hit_10, width, label='Hit@10', color='c', hatch='.')
hit_20 = ax.bar(x + width, hit_20, width, label='Hit@20', color='m', hatch='o')
mrr = ax.bar(x + 2 * width, mrr, width, label='MRR', color='b', hatch='*')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(labels)
legend = ax.legend(loc='best', shadow=True, fontsize='small')
ax.set_ylim(top=0.6)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(hit_1)
autolabel(hit_3)
autolabel(hit_5)
autolabel(hit_10)
autolabel(hit_20)
autolabel(mrr)

plt.show()