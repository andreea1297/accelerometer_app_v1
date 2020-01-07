from model import Net
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import ToTensor, FallDataset
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch.onnx

parser = argparse.ArgumentParser(
    description='Accelerometer.')
parser.add_argument('--csv_file_test', default='test.csv', type=str,
                    help='Specifies the data file for test.')
parser.add_argument('--model_name', default='log/model_20000.tar', type=str,
                    help='Specifies the model.')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Mini batch size.')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for data loading.')
parser.add_argument('--iterations', default=4000, type=int, help='Number of training iterations.')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum.')
parser.add_argument('--gpus', default='0', type=str,
                    help='GPUs to use (ex. ''0, 1, 2, 3''). If None, use CPU.')
args = parser.parse_args()

csv_file_test = args.csv_file_test
num_workers = args.num_workers
batch_size = args.batch_size
iterations = args.iterations
lr = args.learning_rate
momentum = args.momentum
model_name = args.model_name

gpus = args.gpus
use_cuda = False
ngpus = 0
#if gpus is not None:
 #   use_cuda = True
  #  ngpus = len(gpus.split(','))

seed = 16
torch.manual_seed(seed)
#if use_cuda:
 #   device = "cuda"
  #  os.environ['CUDA_VISIBLE_DEVICES'] = gpus
   # torch.cuda.manual_seed(seed)


transformed_dataset_val = FallDataset(csv_file=csv_file_test, transform=transforms.Compose([
                                         ToTensor()
                                     ]))

dataloader_val = DataLoader(transformed_dataset_val, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)

net = Net()

outfile = open('test/out_test.txt', "w")


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# load the model
checkpoint = torch.load(model_name)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

test_loss = 0
correct = 0
total =0

for i0, data in enumerate(dataloader_val):
    data, target = data['data'],data['target']

    # For ONNX. Don't use cuda when generate onnx.
    # torch.onnx.export(net,data,"model.onnx")

    #if use_cuda:
     #   data = data.cuda()
      #  target = target.cuda()
    output = net.forward(data)
    target = target.long()
    test_loss += criterion(output, target).item() 
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    total += batch_size
    if int(target) == 1:
        target = 'sitting'
    else:
        if int(target) == 2:
            target = 'walking'
        else:
            if int(target) == 3:
                target = 'jogging'
            else:
                if int(target) == 6:
                    target = 'standing'
    
    if int(pred) == 1:
        pred = 'sitting'
    else:
        if int(pred) == 2:
            pred = 'walking'
        else:
            if int(pred) == 3:
                pred = 'jogging'
            else:
                if int(pred) == 6:
                    pred = 'standing'
    
    message = 'Expected: {}, Predict: {}'.format(target, pred)
    print (message)
    outfile.write(message+'\n')

test_loss /= total
print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct,total,
                100. * correct / total))