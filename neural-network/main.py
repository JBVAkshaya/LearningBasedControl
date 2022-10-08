from numpy.core.fromnumeric import shape
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_mse(err):
    mse = np.sum(np.square(err))/(2.0*err.shape[0])
    return mse

def compute_acc(pred, gt):
    pred = (pred == pred.max(axis=1)[:,None]).astype(int)
    acc = np.sum(np.multiply(pred, gt))
    return acc/gt.shape[0]
def compute_delta_weight(eta, delta, x):
    
    delta_wt = np.zeros((x.shape[1],delta.shape[1]),dtype=np.float64)
    for k in range (0,x.shape[0]):
        for i in range (0,delta_wt.shape[0]):
            for j in range(0,delta_wt.shape[1]):
                delta_wt[i][j] = delta_wt[i][j] + (eta*delta[k][j]*x[k][i])
    return delta_wt

def update_weight(wt, delta_wt):
    return np.add(wt,delta_wt)

def out_delta(e, y):
    delta = np.multiply(e, np.multiply(y, 1-y))
    return delta

def hid_delta(w, delt, h):
    delta = []
    for x in delt:
        tmp = []
        for y in w:
            tmp.append((x[0]*y[0])+(x[1]*y[1]))
        delta.append(tmp)
    delta = np.array(delta)
    delta = np.multiply(delta[:,:-1], np.multiply(h,1-h))
    return delta

def test (x_test,w1,w2):
    inp = x_test[:,0:5]
    gt = x_test[:,5:]
    
    inp = np.append(inp,np.ones((inp.shape[0],1)),axis=1)
    hidden_layer = 1/(1+np.exp(-np.matmul(inp,w1)))
    hl_with_bias = np.append(hidden_layer,np.ones((hidden_layer.shape[0],1)),axis=1)
    pred = 1/(1+np.exp(-np.matmul(hl_with_bias,w2)))
    err = np.subtract(gt,pred)

    loss = compute_mse(err)
    acc = compute_acc(pred, gt)

    return loss, acc

def train (x_train, w1, w2, eta):
    inp = x_train[:,0:5]
    gt = x_train[:,5:]
    
    inp = np.append(inp,np.ones((inp.shape[0],1)),axis=1)
    hidden_layer = 1/(1+np.exp(-np.matmul(inp,w1)))
    hl_with_bias = np.append(hidden_layer,np.ones((hidden_layer.shape[0],1)),axis=1)
    pred = 1/(1+np.exp(-np.matmul(hl_with_bias,w2)))
    err = np.subtract(gt,pred)
    out_delt = out_delta(err, pred)

    hid_delt = hid_delta(w2,out_delt,hidden_layer)

    delta_w1 = compute_delta_weight(eta, hid_delt, inp)
    w1 = update_weight(w1,delta_w1)

    delta_w2 = compute_delta_weight(eta, out_delt, hl_with_bias)

    w2 = update_weight(w2,delta_w2)

    loss = compute_mse(err)
    acc = compute_acc(pred, gt)
    return w1, w2, loss, acc

if __name__=="__main__":

    # Impact of number of hidden units on training performance.
    df = np.array(pd.read_csv("hw1_data/train1.csv"))

    eta = 0.01
    num_epoch = 400
    x_train ,x_test = train_test_split(df,test_size=0.2)

    num_hiddens = np.arange(1,20,1) 
    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []
    train_loss = 0
    val_loss = 0
    train_acc = 0
    val_acc = 0
    for num_hid in num_hiddens:
        print("number of hidden layers: %d" %(num_hid))
        w1 = np.random.rand(6,num_hid)
        w2 = np.random.rand(num_hid+1,2)
        for epoch in range(0, num_epoch):
            val_loss, val_acc = test(x_test, w1, w2)
            w1, w2, train_loss, train_acc = train(x_train, w1, w2, eta)
            
            print("Epoch %d: train_loss: %f val_loss: %f train_acc: %f val_acc: %f " % (epoch,train_loss, val_loss, train_acc, val_acc))
        t_loss.append(train_loss)
        t_acc.append(train_acc)
        v_loss.append(val_loss)
        v_acc.append(val_acc)
    
    ep = num_hiddens

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6),sharex=True)
    ax1.plot(ep, t_loss, 'b', label='training loss', lw=1) # points
    ax1.plot(ep, v_loss, 'g', label='validation loss', lw=1)
    ax1.legend()

    ax2.plot(ep, t_acc, 'b', label='training accuracy', lw=1) # points
    ax2.plot(ep, v_acc, 'g', label='validation accuracy', lw=1)
    ax2.legend()

    ax2.set_xlabel('number hidden units')
    ax1.set_xlabel('training loss')
    ax2.set_xlabel('training accuracy')

    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.savefig('varrying_hidden_units_t1.png',dpi=300,bbox_inches='tight')
    plt.close('all')
    
    # Impact of training time on training the network
    df = np.array(pd.read_csv("hw1_data/train2.csv"))

    eta = 0.01
    num_hid = 4
    x_train ,x_test = train_test_split(df,test_size=0.2)

    num_epochs = np.arange(10,800,10)
    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []
    train_loss = 0
    val_loss = 0
    train_acc = 0
    val_acc = 0
    for num_epoch in num_epochs:
        # print("number of hidden layers: %d" %(num_hid))
        w1 = np.random.rand(6,num_hid)
        w2 = np.random.rand(num_hid+1,2)
        for epoch in range(0, num_epoch):
            val_loss, val_acc = test(x_test, w1, w2)
            w1, w2, train_loss, train_acc = train(x_train, w1, w2, eta)
            
            print("Epoch %d: train_loss: %f val_loss: %f train_acc: %f val_acc: %f " % (epoch,train_loss, val_loss, train_acc, val_acc))
        t_loss.append(train_loss)
        t_acc.append(train_acc)
        v_loss.append(val_loss)
        v_acc.append(val_acc)

    
    ep = num_epochs

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6),sharex=True)
    ax1.plot(ep, t_loss, 'b', label='training loss', lw=1) # points
    ax1.plot(ep, v_loss, 'g', label='validation loss', lw=1)
    ax1.legend()

    ax2.plot(ep, t_acc, 'b', label='training accuracy', lw=1) # points
    ax2.plot(ep, v_acc, 'g', label='validation accuracy', lw=1)
    ax2.legend()

    ax2.set_xlabel('number of epochs')
    ax1.set_xlabel('training loss')
    ax2.set_xlabel('training accuracy')

    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.savefig('varrying_time_t2.png',dpi=300,bbox_inches='tight')
    plt.close('all')

    # Impact on varying learning rate

    df = np.array(pd.read_csv("hw1_data/train1.csv"))
    
    num_epoch = 500
    num_hid = 4
    x_train ,x_test = train_test_split(df,test_size=0.2)

    etas = np.arange(0.001,0.02,0.001) 
    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []
    train_loss = 0
    val_loss = 0
    train_acc = 0
    val_acc = 0
    for eta in etas:
        print("number of hidden layers: %d" %(num_hid))
        w1 = np.random.rand(6,num_hid)
        w2 = np.random.rand(num_hid+1,2)
        for epoch in range(0, num_epoch):
            val_loss, val_acc = test(x_test, w1, w2)
            w1, w2, train_loss, train_acc = train(x_train, w1, w2, eta)
            
            print("Epoch %d: train_loss: %f val_loss: %f train_acc: %f val_acc: %f " % (epoch,train_loss, val_loss, train_acc, val_acc))
        t_loss.append(train_loss)
        t_acc.append(train_acc)
        v_loss.append(val_loss)
        v_acc.append(val_acc)
    
    ep = etas

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6),sharex=True)
    ax1.plot(ep, t_loss, 'b', label='training loss', lw=1) # points
    ax1.plot(ep, v_loss, 'g', label='validation loss', lw=1)
    ax1.legend()

    ax2.plot(ep, t_acc, 'b', label='training accuracy', lw=1) # points
    ax2.plot(ep, v_acc, 'g', label='validation accuracy', lw=1)
    ax2.legend()

    ax2.set_xlabel('learning rate')
    ax1.set_xlabel('training loss')
    ax2.set_xlabel('training accuracy')

    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.savefig('varrying_learning_rate_001_to_02_t1.png',dpi=300,bbox_inches='tight')
    plt.close('all')

    # Performance Evaluation:
    df = np.array(pd.read_csv("hw1_data/train2.csv"))
    
    num_epoch = 500
    num_hid = 4
    x_train ,x_test = train_test_split(df,test_size=0.2)

    eta = 0.01
    
    
    print("number of hidden layers: %d" %(num_hid))
    w1 = np.random.rand(6,num_hid)
    w2 = np.random.rand(num_hid+1,2)
    for epoch in range(0, num_epoch):
        val_loss, val_acc = test(x_test, w1, w2)
        w1, w2, train_loss, train_acc = train(x_train, w1, w2, eta)
        
        print("Epoch %d: train_loss: %f val_loss: %f train_acc: %f val_acc: %f " % (epoch,train_loss, val_loss, train_acc, val_acc))
    
    t1_data = np.array(pd.read_csv("hw1_data/test1.csv"))
    t_loss_1, t_acc_1 =  test(t1_data, w1, w2)   

    t2_data = np.array(pd.read_csv("hw1_data/test2.csv"))
    t_loss_2, t_acc_2 =  test(t2_data, w1, w2)

    t3_data = np.array(pd.read_csv("hw1_data/test3.csv"))
    t_loss_3, t_acc_3 =  test(t3_data, w1, w2)

    print("test1: loss: %f acc: %f" %(t_loss_1, t_acc_1))
    
    print("test2: loss: %f acc: %f" %(t_loss_2, t_acc_2))

    print("test3: loss: %f acc: %f" %(t_loss_3, t_acc_3))