import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader, ArrayDataset
import matplotlib.pyplot as plt

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

### Get data from files
def get_data():    
    data = pd.read_csv("2.csv", header=0)
    y, X = np.array(data.iloc[:,-1]), np.array(data.iloc[:,0:-1])
    print(X.shape); print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, random_state=0)

    return X_train, X_test, y_train, y_test
X_tr, X_te, y_tr, y_te = get_data()



np.random.seed(11)

N=X_tr.shape[0]
p=X_tr.shape[1]

X_train=nd.array(X_tr)
y_train=nd.array(y_tr)

batch_size=200
train_dataset = ArrayDataset(X_train, y_train)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
X_test=nd.array(X_te).as_in_context(ctx)
y_test=nd.array(y_te).as_in_context(ctx)


net=nn.Sequential()
net.add(nn.Dense(1000,activation='relu'),nn.Dense(1, activation=None))


net.initialize(mx.init.Xavier(), ctx=ctx)

for i, (data, label) in enumerate(train_data):
    aa=net(data.as_in_context(ctx))
    break
net_initial=net


def init_masks_percents(net,p):
    masks={}
    percents={}
    for i in enumerate(net):
        masks[i[0]]=nd.ones(net[i[0]].weight.data().shape).as_in_context(ctx)
        percents[i[0]]=p
    return masks,percents
        


def get_weights(net):
    weights={}
    for i in enumerate(net):
        weights[i[0]]=net[i[0]].weight.data()
    return weights



def prune_by_percent(percents, masks, final_weights):
    def prune_by_percent_once(percent, mask, final_weight):
        sorted_weights = np.sort(np.abs(final_weight.asnumpy()[mask.asnumpy() == 1]))
        cutoff_index = np.round(percent * sorted_weights.size).astype(int)
        cutoff = sorted_weights[cutoff_index]
        return nd.where(nd.abs(final_weight) <= cutoff, nd.zeros(mask.shape).as_in_context(ctx), mask)
    
    new_masks = {}
    for k, percent in percents.items():
        new_masks[k] = prune_by_percent_once(percent, masks[k], final_weights[k])
    return new_masks



class MaskedNet(nn.Block):
    # For FC network
    def __init__(self,net,masks,**kwargs):
        super(MaskedNet, self).__init__(**kwargs)  
        self.net=net
        self.masks=masks

    def forward(self, x):
        x=x.T
        for i in enumerate(self.net):
            x=nd.dot(self.net[i[0]].weight.data()*self.masks[i[0]],x)
            x=x+self.net[i[0]].bias.data().reshape((self.net[i[0]].bias.data().size,1))
            if i[0]< len(self.masks)-1:
                x=nd.relu(x)
            else:
                break
        return x.T

### initialize Masknet    
masks,percents=init_masks_percents(net,0.001)
maskednet=MaskedNet(net_initial,masks)


trainer = gluon.Trainer(
    params=maskednet.collect_params(),
    optimizer = 'adam',

)

metric = mx.metric.Accuracy()
loss_function = gluon.loss.L2Loss() 


epochs = 400
num_batches = N / batch_size
trainerr=[]
testerr=[]
ep=[]        
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        #print(i)
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = maskednet(data)
            loss=loss_function(output, label)
        loss.backward()
      
        trainer.step(batch_size)#,ignore_stale_grad=True) ###ignore
        cumulative_loss += nd.mean(loss).asscalar()
    
    if e%5==0 and e!=0:
        
        
        final_w=get_weights(maskednet.net)        
        masks=prune_by_percent(percents, masks, final_w)  # Update Masks
        maskednet=MaskedNet(net_initial,masks)            # Reset Network with new mask
        print("Epoch %s, loss: " % (e))
        f_loss=nd.mean(loss).asscalar()
        testloss=loss_function(net(nd.array(X_test).as_in_context(ctx)),y_test)
        print('train_loss:',f_loss,'test_loss:',nd.mean(testloss).asnumpy())
        trainerr.append(f_loss)
        testerr.append(nd.mean(testloss).asnumpy())
        ep.append(e)
        
plt.plot(ep,trainerr)
plt.plot(ep,testerr)
plt.show() 
