from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
import os
sys.path.insert(0,os.path.join(os.getcwd(),'auxiliary'))
#sys.path.append('./auxiliary/')
from dataset import *
from model import *
from my_utils import *
from ply import *
import os
import json
import time, datetime
import visdom

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives in the atlas')
parser.add_argument('--super_points', type=int, default = 2500,  help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="AE_AtlasNet"   ,  help='visdom environment')
parser.add_argument('--accelerated_chamfer', type=int, default =0   ,  help='use custom build accelarated chamfer')

opt = parser.parse_args()
print (opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
if opt.accelerated_chamfer:
    sys.path.append("./extension/")
    import dist_chamfer as ext
    distChamfer =  ext.chamferDist()

else:
    def pairwise_dist(x, y):
        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        P = (rx.t() + ry - 2*zz)
        return P


    def NN_loss(x, y, dim=0):
        dist = pairwise_dist(x, y)
        values, indices = dist.min(dim=dim)
        return values.mean()


    def distChamfer(a,b):
        x,y = a,b
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2,1))
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2,1) + ry - 2*zz)
        return torch.min(P, 1)[0], torch.min(P, 2)[0], torch.min(P, 1)[1], torch.min(P, 2)[1]
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name =  os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10
# ========================================================== #


# ===================CREATE DATASET================================= #
#Create train/test dataloader
dataset = ShapeNet( normal = False, class_choice = None, train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet( normal = False, class_choice = None, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
network = AE_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network.cuda() #put network on GPU
network.apply(weights_init) #initialization of the weight

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001 #learning rate
optimizer = optim.Adam(network.parameters(), lr = lrate)
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
#meters to record stats on learning
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')

#initialize learning curve on visdom, and color for each primitive in visdom display
train_curve = []
val_curve = []
labels_generated_points = torch.Tensor(range(1, (opt.nb_primitives+1)*(opt.num_points//opt.nb_primitives)+1)).view(opt.num_points//opt.nb_primitives,(opt.nb_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.nb_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
# ========================================================== #

# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_loss.reset()
    network.train()
    
    # learning rate schedule
    if epoch==100:
        optimizer = optim.Adam(network.parameters(), lr = lrate/10.0)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        img, points, cat, _, _ = data
        points = points.transpose(2,1).contiguous()
        points = points.cuda()
        #SUPER_RESOLUTION optionally reduce the size of the points fed to PointNet
        points = points[:,:,:opt.super_points].contiguous()
        #END SUPER RESOLUTION
        pointsReconstructed  = network(points) #forward pass
        dist1, dist2,_, _ = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed) #loss function
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net.backward()
        train_loss.update(loss_net.item())
        optimizer.step() #gradient update

        print('[%d: %d/%d] train loss:  %f ' %(epoch, i, len_dataset/32, loss_net.item()))


    #UPDATE CURVES
    train_curve.append(train_loss.avg)

    # VALIDATION
    val_loss.reset()
    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset()

    network.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            img, points, cat, _ , _ = data
            points = points.transpose(2,1).contiguous()
            points = points.cuda()
            #SUPER_RESOLUTION
            points = points[:,:,:opt.super_points].contiguous()
            #END SUPER RESOLUTION
            pointsReconstructed  = network(points)
            dist1, dist2,_,_ = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
            val_loss.update(loss_net.item())
            dataset_test.perCatValueMeter[cat[0]].update(loss_net.item())
            print('[%d: %d/%d] val loss:  %f ' %(epoch, i, len(dataset_test), loss_net.item()))

        #UPDATE CURVES
        val_curve.append(val_loss.avg)


    #dump stats in log file
    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "super_points" : opt.super_points,
      "bestval" : best_val_loss,

    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')

    #save last network
    print('saving net...')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
