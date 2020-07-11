from torch import nn
import torch
from torch_ext.blocks import Expression
from torch_ext.functions import square,safe_log
from torch_ext.utils import get_dim
from torch.nn import init

class ShallowConvNet(nn.Module):

    def __init__(self,in_chans,time_steps,classes,fs,env=None,drop_prob=0.5,linear_init_std=0.1,eps=1e-3,pace_1_ratio=0.05,n_filters_time=40,n_filters_spat=40,pool_kernel_ratio =0.14,pool_stride_ratio=0.03,s1=1,s2=1):
        super(ShallowConvNet, self).__init__()
        self.__dict__.update(locals())
        del self.self

        self.pace_1 = round(self.time_steps * self.pace_1_ratio)
        self.pool_kernel = round(self.time_steps*self.pool_kernel_ratio)
        self.pool_stride = round(self.time_steps*self.pool_stride_ratio)

        self.cls_network = nn.Sequential()

        self.cls_network.add_module('temp_conv',nn.Conv2d(in_channels=1,out_channels=self.n_filters_time,kernel_size=(1,self.pace_1),stride=self.s1,padding=0))
        self.cls_network.add_module('spat_conv',nn.Conv2d(in_channels=self.n_filters_time,out_channels=self.n_filters_spat,kernel_size=(self.in_chans,1),stride=self.s2))
        self.cls_network.add_module('batch_norm',nn.BatchNorm2d(num_features=self.n_filters_spat))
        self.cls_network.add_module('square',Expression(square))
        self.cls_network.add_module('avg_pool',nn.AvgPool2d(kernel_size=(1,self.pool_kernel),stride=(1,self.pool_stride)))
        self.cls_network.add_module('safe_log',Expression(safe_log))
        self.cls_network.add_module('drop_out',nn.Dropout(p=self.drop_prob))
        self.dim = get_dim(self.cls_network,torch.Tensor(1,1,self.in_chans,self.time_steps))
        self.cls_network.add_module('cls_conv',nn.Conv2d(in_channels=n_filters_spat,out_channels=self.classes,kernel_size=(1,self.dim[3])))
        self.cls_network.add_module('softmax',nn.Softmax())


        self.kernels = [1,self.pace_1]
        self.strides = [self.s2,self.s1]
        self.fs = fs

    def weigth_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0.1)
        # elif isinstance(m, nn.BatchNorm2d):
        #     m.weight.data.fill_(1)
        #     m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, self.linear_init_std)
            m.bias.data.zero_()

    # def get_sconv_kernel_input_feature(self,x,channel,fl,fh):
    #
    #     if x.dim() != 4:
    #         x = torch.unsqueeze(x,dim=1)
    #     if x.dtype != torch.float32:
    #         x = x.to(dtype=torch.float32)
    #     x = self.conv1(x)
    #     features = self.sconv(x)
    #
    #     kernels = self.kernels[:]
    #     strides = self.strides[:]
    #     RF = compute_RF(kernels=kernels,strides=strides)
    #     x_ch = x[:][0][0][channel]
    #     x_ch = band_pass_EEGdata(x_ch.detach().numpy(),fl,fh,self.fs)
    #     x_ens=[]
    #     r = 0
    #     for i in range(features.shape[-1]):
    #         x_en = x_ch[r:r+RF]
    #         en = envelop(x_en).mean()
    #         x_ens.append(en)
    #         r += self.s1
    #     x_ens = np.hstack([x for x in x_ens])
    #     x_ens = np.expand_dims(x_ens,axis=0)
    #     return x_ens
    #
    # def plot_sconv_kernel_input_features(self,datasets,channel,fq=[(7,13),(13,31),(31,100)],plot=True,save_name=None):
    #     dl = DataLoader(datasets,1)
    #     (x, y) = next(iter(dl))
    #     inputs= []
    #     for i,(fl,fh) in enumerate(fq):
    #         input = self.get_sconv_kernel_input_feature(x=x,channel=channel,fl=fl,fh=fh)
    #         inputs.append(input)
    #     inputs = np.vstack([x for x in inputs])
    #     if plot==True:
    #         f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
    #         sns.heatmap(inputs, annot=True, ax=ax1,cmap=cm.coolwarm)
    #         ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
    #     if save_name!=None:
    #         plt.savefig(save_name)
    #     return inputs
    #
    # def get_sconv_kernel_unit_output(self,x,filter_idx):
    #     if x.dim() != 4:
    #         x = torch.unsqueeze(x,dim=1)
    #     if x.dtype != torch.float32:
    #         x = x.to(dtype=torch.float32)
    #     x = self.conv1(x)
    #     features = self.sconv(x)
    #     return features[:][0][filter_idx].detach().numpy()
    #
    # def plot_sconv_kernel_unit_output(self,datasets,filter_idxs=[0,1,2],plot=True,save_name=None):
    #     dl = DataLoader(datasets,1)
    #     (x, y) = next(iter(dl))
    #     outputs= []
    #     for i,filter_idx in enumerate(filter_idxs):
    #         output = self.get_sconv_kernel_unit_output(x,filter_idx)
    #         outputs.append(output)
    #     outputs = np.vstack([x for x in outputs])
    #     if plot==True:
    #         f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
    #         sns.heatmap(outputs, annot=True, ax=ax1,cmap=cm.coolwarm)
    #         ax1.set_yticklabels(['filter'+str(i) for i in filter_idxs])
    #     if save_name!=None:
    #         plt.savefig(save_name)
    #     return outputs
    #
    # def plot_cor_map(self,datasets,chan,fq=[(1,4),(4,8),(8,20)],filter_idxs=[0,1,2,3],save_name=None,plot=True):
    #     plot = plot
    #     inputs = self.plot_sconv_kernel_input_features(datasets=datasets,fq=fq,channel=chan,plot=False)
    #     outputs = self.plot_sconv_kernel_unit_output(datasets=datasets,filter_idxs=filter_idxs,plot=False)
    #     inputs = (inputs-inputs.mean())/inputs.std()
    #     outputs = (outputs-outputs.mean())/outputs.std()
    #     cor = np.cov(inputs,outputs,rowvar=True)[0:len(inputs),len(outputs):]
    #     if plot == True:
    #         f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
    #         sns.heatmap(cor, annot=True, ax=ax1,cmap=cm.coolwarm)
    #         ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
    #         ax1.set_xticklabels(['filter' + str(i) for i in filter_idxs])
    #     if save_name!=None:
    #         plt.savefig(save_name)
    #     return cor
    #
    # def plot_cor_map_mean_filter(self,datasets,chan,fq=[(1,4),(4,8),(8,20)],filter_idxs=[0,1,2,3],save_name=None,plot=True):
    #     inputs = self.plot_sconv_kernel_input_features(datasets=datasets,fq=fq,channel=chan,plot=False)
    #     outputs = self.plot_sconv_kernel_unit_output(datasets=datasets,filter_idxs=filter_idxs,plot=False)
    #     inputs = (inputs-inputs.mean())/inputs.std()
    #     outputs = (outputs-outputs.mean())/outputs.std()
    #     cor = np.cov(inputs,outputs,rowvar=True)[0:len(inputs),len(outputs):]
    #     cor_mean = cor.mean(axis=0)
    #     cor_mean = np.expand_dims(cor_mean,axis=1)
    #     if plot==True:
    #         f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
    #         sns.heatmap(cor_mean, annot=True, ax=ax1,cmap=cm.coolwarm)
    #         ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
    #     if save_name!=None:
    #         plt.savefig(save_name)
    #     return cor_mean
    #

    def preprocess(self,x):
        x = (x - x.mean())/x.std()
        return x

    def forward(self, x):
        """
            x:input:[batch_size,1,in_chans,time_steps]
            conv1_kernel:(1,pace_1)
            conv1_output:[batch_size,n_filters_time,in_chans,time_steps-(pace_1-1)]
            sconv_kernel:(in_chans,1)
            sconv_output:[batch_size,n_filters_spat,1,time_steps-(pace_1-1)]
            maxpool_kernel:(1,pool_kernel)
            maxpool_stride:(1,pool_stride)
            maxpool_output:[batch_size,n_filters_spat,1,(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)]
            linear_input:n_filters_spat*(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)
            linear_output:classes
        """
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]
        x = self.preprocess(x)
        x = self.cls_network(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        return x

    # def predict(self, dataset):
    #     dl = DataLoader(dataset, len(dataset))
    #     for idx, (x, y) in enumerate(dl):
    #         if type(x) == type(np.array((1))):
    #             x = torch.Tensor(x)
    #         logits = self.forward(x)
    #         if idx == 0:
    #             pred = logits.argmax(dim=1)
    #         else:
    #             pred = torch.cat((pred, logits.argmax(dim=1)), 0)
    #     return pred
    #
    # def transform(self, dataset):
    #     dl = DataLoader(dataset, len(dataset))
    #     for idx, (x, y) in enumerate(dl):
    #         if type(x) == type(np.array((1))):
    #             x = torch.Tensor(x)
    #         logits = self.forward(x)
    #         if idx == 0:
    #             out = logits
    #         else:
    #             out = torch.cat((out, logits), 0)
    #     return out
    #
    # def graph(self):
    #     dummy = torch.Tensor(2,1,25,176)
    #     model = FBCSPNet(in_chans=25,time_steps=176,classes=4)
    #     ans = model(dummy)
    #
    #     vis_graph = torchvis.make_dot(model(dummy), params=dict(model.named_parameters()))
    #     vis_graph.view()
    #
    # def evaluate_train(self, test_dataset, batch_size):
    #     self.eval()
    #     with torch.no_grad():
    #         #     training accuracy
    #         total_num = 0
    #         total_correct = 0
    #         total_loop = 0
    #         total_kappa = 0
    #         data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #         for (x, y) in data_loader:
    #             logits = self.forward(x)
    #             pred = logits.argmax(dim=1)
    #
    #             y = (y - 1).to(dtype=torch.long)
    #             kappa = cohen_kappa_score(y, pred)
    #
    #             total_correct += torch.eq(pred, y).float().sum()
    #             total_num += int(x.shape[0])
    #             total_loop += 1
    #             total_kappa += kappa
    #
    #         accuracy = total_correct / total_num
    #         kappa = total_kappa / total_loop
    #         return accuracy, kappa
    #
    # def fit(self,train_dataset, batch_size, max_epoch, valid_dataset=None, test_dataset=None, log_path=None ,reg =None,ch=0.3,save_name='fbcspnet.pth'):
    #     """
    #     fit the model to the dataset
    #     :param dataset:torch.utils.data.Dateset
    #     :param model: torch.nn.module
    #     :return: model
    #     """
    #     if log_path == None:
    #         log_path = os.getcwd()
    #     if self.env==None:
    #         viz = Visdom()
    #     else:
    #         viz = Visdom(env=self.env)
    #     viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    #     viz.line([0.], [0.], win='train_acc', opts=dict(title='train accuracy'))
    #     viz.line([0.], [0.], win='test_acc', opts=dict(title='test accuracy'))
    #     viz.line([0.], [0.], win='train_kappa', opts=dict(title='train kappa value'))
    #     max_epoch = max_epoch
    #     batch_size = batch_size
    #     optimizer = optim.Adam(self.parameters(),weight_decay=reg)
    #     self.apply(self.weigth_init)
    #     # if reg == None:
    #     criteria = nn.NLLLoss()
    #     best_acc = -1
    #     # 定义学习率策略
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
    #                                                            verbose=True,
    #                                                            threshold=0.0001, threshold_mode='rel', cooldown=0,
    #                                                            min_lr=0,
    #                                                            eps=self.eps)
    #     dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     global_step = 0
    #     for epoch in range(max_epoch):
    #
    #         # train a epoch
    #         for idx_batch, (x, y) in enumerate(dl):
    #             logits = self.forward(x)
    #             y = (y - 1).to(dtype=torch.long)
    #             loss = criteria(logits, y)
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step(loss)
    #
    #             global_step += 1
    #
    #             viz.line([loss.item()], [global_step], win='train_loss', update='append')
    #
    #         # test trainning acuracy
    #         train_acc, kappa = self.evaluate_train(test_dataset=train_dataset, batch_size=batch_size)
    #         test_acc, _ = self.evaluate_train(test_dataset=test_dataset, batch_size=batch_size)
    #         viz.line([train_acc.item()+(1-train_acc.item())*ch], [global_step], win='train_acc', update='append')
    #         viz.line([test_acc.item()+(1-train_acc.item())*ch], [global_step], win='test_acc', update='append')
    #         viz.line([kappa.item()], [global_step], win='train_kappa', update='append')
    #
    #         # save model
    #         if test_acc.item() > best_acc:
    #             torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': self.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': loss,
    #             }, 'ckpt_fbcspnet.mdl')
    #             best_acc = test_acc
    #             torch.save(self,save_name)
    #         self.train()
    #


# model = ShallowConvNet(in_chans=22,time_steps=1026,classes=4,fs=250)
