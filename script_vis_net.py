import scalp
from envelope import envelop
from BCICdatasets import BCIC4_2a
from BCI_database import Subject_dataset
import scipy.signal as Signal
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from modules import FBCSPNet,band_pass_EEGdata
import torch
import seaborn as sns
data_42a =BCIC4_2a()
data_42a.load_data()
data_42a_1 = Subject_dataset(database=data_42a,subject_id=1)

dl = DataLoader(data_42a_1)
class_dict= {1:'cueOnsetLeft',2:'cueOnsetRight',3: 'cueOnsetTongue', 4:'cueOnsetFoot'}

#draw scalp envelope
x_tp,y_tp =next(iter(dl))
vmin = float(x_tp.min().numpy())
vmax = float(x_tp.max().numpy())
f = 0
for i in range(20):
    f = f + 2
    ch_datas = []
    sos = Signal.cheby2(12, 20, [f, f+2], 'bp', fs=250, output='sos')
    for j,(x,y) in enumerate(dl):
        if j<1:
            x = x[0].numpy()
            for ch_data in x:
                ch_data = Signal.sosfilt(sos,ch_data)
                en_array = envelop(ch_data.transpose())
                en_array = en_array.mean()
                ch_datas.append(en_array)
    values = np.array([ch_datas]).transpose().squeeze(1)
    names = data_42a.names
    #scalp.plot_scalp(values,names,dataset='BCIC4_2a',norm=Normalize(vmin,vmax))
    scalp.plot_scalp(values, names, dataset='BCIC4_2a')
    #scalp.ax_scalp(values,names,dataset='BCIC4_2a')
    name = str(f)+'-'+str(f+2)+class_dict[int(y[0].numpy())]
    plt.savefig(name)
    plt.close()


#可视化网络
dl = DataLoader(data_42a_1,batch_size=16)

x_tp,y_tp =next(iter(dl))

dummy = x_tp
model = FBCSPNet(in_chans=22,time_steps=176,classes=4,fs=data_42a.fs)
inputs = []
outputs = []
filters = [0,1,2]
fq = [(1,4),(4,8),(8,20)]
for i,f in enumerate(filters):
    dummy_i = dummy
    input = model.get_sconv_kernel_input_feature(dummy_i,channel=0,fl=fq[i][0],fh=fq[i][1])
    output = model.get_sconv_kernel_unit_output(dummy,f)
    inputs.append(input)
    outputs.append(output)
inputs = np.vstack([x for x in inputs])
outputs = np.vstack([x for x in outputs])
# cor = np.matmul(inputs,(outputs/outputs.std()).transpose())
cor = np.cov(inputs,outputs)[0:3,3:6]


f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
sns.heatmap(cor, annot=True, ax=ax1)
ax1.set_xticklabels(['filter1','filter2','filter3'])
ax1.set_yticklabels(['4-8Hz','8-12Hz','12-16Hz'])

