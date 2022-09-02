"""
===============
Pyplot tutorial
===============

An introduction to the pyplot interface.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

width = .35 # width of a bar

m1_t = pd.DataFrame({
 'MRR' : [0.276, 0.322, 0.348],
 '#Triples' : [19651, 32302, 37140]
 })


# m2_t = pd.DataFrame({
#  'MRR' : [0.282, 0.333, 0.343],
#  '#Triples' : [19651, 37140, 32302 ]
#  })

m1_t[['#Triples']].plot(kind='bar', width = width )
m1_t['MRR'].plot(secondary_y=True, color='r')

ax = plt.gca()
plt.xlim([-width, len(m1_t['MRR'])-width])
ax.set_xticklabels(('Agent A', 'Agent B', 'Agent C'))

plt.show()