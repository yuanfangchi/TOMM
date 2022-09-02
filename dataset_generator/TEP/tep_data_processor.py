import pandas as pd
import matplotlib.pyplot as plt
import random

# Loading data
df_fault = pd.read_csv('TEP_Faulty_Training.csv', index_col=0)
df_fault.head()

df_faultFree = pd.read_csv('TEP_FaultFree_Training.csv', index_col=0)
df_faultFree.head()

X_dict = {
'XMEAS_1':'A_feed_stream',
'XMEAS_2':'D_feed_stream',
'XMEAS_3':'E_feed_stream',
'XMEAS_4':'Total_fresh_feed_stripper',
'XMEAS_5':'Recycle_flow_into_rxtr',
'XMEAS_6':'Reactor_feed_rate',
'XMEAS_7':'Reactor_pressure',
'XMEAS_8':'Reactor_level',
'XMEAS_9':'Reactor_temp',
'XMEAS_10':'Purge_rate',
'XMEAS_11':'Separator_temp',
'XMEAS_12':'Separator_level',
'XMEAS_13':'Separator_pressure',
'XMEAS_14':'Separator_underflow',
'XMEAS_15':'Stripper_level',
'XMEAS_16':'Stripper_pressure',
'XMEAS_17':'Stripper_underflow',
'XMEAS_18':'Stripper_temperature',
'XMEAS_19':'Stripper_steam_flow',
'XMEAS_20':'Compressor_work',
'XMEAS_21':'Reactor_cooling_water_outlet_temp',
'XMEAS_22':'Condenser_cooling_water_outlet_temp',
'XMEAS_23':'Composition_of_A_rxtr_feed',
'XMEAS_24':'Composition_of_B_rxtr_feed',
'XMEAS_25':'Composition_of_C_rxtr_feed',
'XMEAS_26':'Composition_of_D_rxtr_feed',
'XMEAS_27':'Composition_of_E_rxtr_feed',
'XMEAS_28':'Composition_of_F_rxtr_feed',
'XMEAS_29':'Composition_of_A_purge',
'XMEAS_30':'Composition_of_B_purge',
'XMEAS_31':'Composition_of_C_purge',
'XMEAS_32':'Composition_of_D_purge',
'XMEAS_33':'Composition_of_E_purge',
'XMEAS_34':'Composition_of_F_purge',
'XMEAS_35':'Composition_of_G_purge',
'XMEAS_36':'Composition_of_H_purge',
'XMEAS_37':'Composition_of_D_product',
'XMEAS_38':'Composition_of_E_product',
'XMEAS_39':'Composition_of_F_product',
'XMEAS_40':'Composition_of_G_product',
'XMEAS_41':'Composition_of_H_product',
'XMV_1':'D_feed_flow_valve',
'XMV_2':'E_feed_flow_valve',
'XMV_3':'A_feed_flow_valve',
'XMV_4':'Total_feed_flow_stripper_valve',
'XMV_5':'Compressor_recycle_valve',
'XMV_6':'Purge_valve',
'XMV_7':'Separator_pot_liquid_flow_valve',
'XMV_8':'Stripper_liquid_product_flow_valve',
'XMV_9':'Stripper_steam_valve',
'XMV_10':'Reactor_cooling_water_flow_valve',
'XMV_11':'Condenser_cooling_water_flow_valve'
#'XMV_12':'Agitator_speed'
   }

x_dict_value_list = list(X_dict.values())

df_faultFree = df_faultFree.rename(columns = lambda x:X_dict[x.upper()] if x.upper() in X_dict.keys()  else x)
df_fault = df_fault.rename(columns = lambda x:X_dict[x.upper()] if x.upper() in X_dict.keys()  else x)

# for run in random.sample(range(1, 184), 10):
tmp_faultFree = df_faultFree[(df_faultFree['simulationRun'] == 2) & (df_faultFree['faultNumber'] == 0)].reset_index()
tmp_fault = df_fault[(df_fault['simulationRun'] == 2) & (df_fault['faultNumber'] == 5)].reset_index()

#for item in X_dict.values():
    #item = 'D_feed_flow_valve'

for i in range(29, 52):
    print(i)
    fig = tmp_faultFree[x_dict_value_list[i]].plot()
    fig = tmp_fault[x_dict_value_list[i]].plot()
    plt.xlabel(x_dict_value_list[i])
    plt.show()

#UPPER_LIMIT = 2724.5
#LOWER_LIMIT = 2685.6

#plt.plot([20, 20], [2600, 2850], '-r')
#plt.plot([0, 500], [UPPER_LIMIT, UPPER_LIMIT], '-k')
#plt.plot([0, 500], [LOWER_LIMIT, LOWER_LIMIT], '-k')

