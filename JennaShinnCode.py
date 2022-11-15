
######## JENNA SHINN ########
######## CODE FROM Q7 #########


# %%
# General Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('cleaned.csv')

# %%
# Create lists of pitchers
pitchers_list = list(df['PitcherId'].unique())

starting_pitchers = df.loc[df['Inning'] == 1]
starters_list = list(starting_pitchers['PitcherId'].unique())

bullpen = df.loc[~df['PitcherId'].isin(starters_list)]
bullpen_list = list(bullpen['PitcherId'].unique())

# %%
#### FUNCTIONS ####
# Returns the number of innings pitched
def innings_pitched(df):
    one_out_plays = ['field_out','force_out','fielders_choice','fielders_choice_out',
                     'sac_bunt', 'sac_fly', 'strikeout', 'caught_stealing_2b']
    two_out_plays = ['grounded_into_double_play', 'double_play']
    three_out_plays = ['triple_play']
    dates = list(df.GamePk.unique())
    ip = 0
    for d in dates:
        max_inning = int(max(df[df.GamePk == d].Inning))
        game_ip = 0
        for i in range(1, (max_inning + 1)) :
            df1 = df[(df.GamePk == d) & (df.Inning == i)]
            one_outs = sum(df1.PitchCall.isin(one_out_plays))
            two_outs = 2 * sum(df1.PitchCall.isin(two_out_plays))
            three_outs = 3 * sum(df1.PitchCall.isin(three_out_plays))
            game_ip += (one_outs + two_outs + three_outs)/3.0
            ip += (one_outs + two_outs + three_outs)/3.0
    return ip

# Calculates FIP
def fip(df):
    hr = sum(df.PitchCall.isin(['home_run']))
    hr = 13*hr
    bb = sum(df.PitchCall.isin(['walk']))
    hbp = sum(df.PitchCall.isin(['hit_by_pitch']))
    bbhbp = 3*(bb+hbp)
    k = sum(df.PitchCall.isin(['strikeout']))
    k = 2*k
    ip = innings_pitched(df)
    cfip = 3.133 # from fangraphs
    fipt = (hr + bbhbp -k)
    fip = (fipt)/ip
    fip = fip + cfip
    return fip

# Calculates WHIP
def whip(df):
    bb = sum(df.PitchCall.isin(['walk']))
    hits = sum(df.PitchCall.isin(['single', 'double','triple','home_run']))
    ip = innings_pitched(df)
    wh = (bb+hits)
    whip = wh/ip
    return whip

# Returns k/9
def strikeout_ratio(df):
    ip = innings_pitched(df)
    ks = sum(df.PitchCall.isin(['strikeout']))
    k_ratio = (ks)/(ip)
    k9_ratio = 9*k_ratio
    return k9_ratio

# Returns bb/9
def walks_ratio(df):
    bb = sum(df.PitchCall.isin(['walk']))
    ip = innings_pitched(df)
    bb9_ratio = (bb/ip)*9
    return bb9_ratio
    

# %%
# Creates dataframe for BullPen stats
pitcher = []
ip_ = []
fip_ = []
whip_ = []
strikeouts_ = []
walks_ = []
    
for p in bullpen_list:
    pitcher_df = df.loc[df['PitcherId'] == p]
    
    pitcher.append(p)
    ip_.append(innings_pitched(pitcher_df))
    fip_.append(fip(pitcher_df))
    whip_.append(whip(pitcher_df))
    strikeouts_.append(strikeout_ratio(pitcher_df))
    walks_.append(walks_ratio(pitcher_df))
    
bp_df= pd.DataFrame({'PitcherID': pitcher, 
        'IP':ip_, 
        'K per 9':strikeouts_,
        'BB per 9':walks_,
        'FIP':fip_,
        'WHIP':whip_}).round(2)


## TABLE FOR BULLPEN STATS
properties = {"border": "2px solid gray", "color": "black", "font-size": "14px", "text-align": "center"}
headers = {"selector": "th:not(.index_name)",
    "props": "background-color: grey; color: white; text-align: center"}

bp_df.style.hide_index().format(precision = 2).background_gradient(cmap = "BuPu", subset=["K per 9"], vmin=0, vmax=50).background_gradient(cmap = "BuPu", subset=["BB per 9"], vmin=0, vmax=25).background_gradient(cmap = "BuPu", subset=["WHIP"], vmin=0, vmax=8).set_properties(**properties).set_table_styles([headers])



# %%
#### Functions for Starters ####
def swingK (pitcher):
    pitcher_df = df.loc[df['PitcherId'] == pitcher]
    swingK = pitcher_df.loc[pitcher_df['PitchCall'] == 'swinging_strike'].groupby('PitchType').agg(SwingK = ('PitchType', 'count'))
    swingK['PitchCount'] =  pitcher_df.groupby('PitchType').agg(PitchCount = ('PitchType', 'count')) 
    swingK['SwingStrike%'] = ((swingK['SwingK']/swingK['PitchCount'])*100).round(2)      
    swingK['PitchType'] = swingK.index
    swingK['Pitcher'] = pitcher
    
    return swingK

# %%
# Starters
pitch1 = swingK(1)
pitch11 = swingK(11)
pitch7 = swingK(7)

starters = pd.concat([pitch1, pitch11, pitch7])

# %%
## Graph for Starting Pitchers
swing_strike = starters['SwingStrike%']

fig, ax = plt.subplots(figsize=(12,6))
sns.set(style='darkgrid')
ax = sns.barplot(data=starters, x="Pitcher", y="SwingStrike%", hue="PitchType", palette = "crest")

ax = plt.gca()

for container in ax.containers:
    ax.bar_label(container, fontsize = 12)
    
yticks = np.arange(0,35,5)
yticklabels = [str(y) + '%' for y in yticks]
plt.yticks(yticks, labels = yticklabels)

plt.legend(title="Pitch Type", bbox_to_anchor=(1.02, 1), fontsize='small', fancybox=True)

plt.xlabel('\nPitcher ID\n')
plt.ylabel('\nSwinging Strike Rate\n')
plt.title('\nSwinging Strike Rate for Starters\n', fontsize=16)

plt.tight_layout()
plt.savefig('starters.svg', dpi = 150)







