import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import ToTensor, FallDataset
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from model import Net
    
# Declare arguments
parser = argparse.ArgumentParser(
    description='Accelerometer')
parser.add_argument('--csv_file_train', default='train.csv', type=str,
                    help='Specifies the data file for training.')
parser.add_argument('--csv_file_valid', default='val.csv', type=str,
                    help='Specifies the data file for validation.')
parser.add_argument('--batch_size', default=32, type=int, 
                    help='Mini batch size.')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for data loading.')
parser.add_argument('--iterations', default=21000, type=int, help='Number of training iterations.')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum.')
parser.add_argument('--val_iter', default=1000, type=int,
                    help='Frequency for validation.')
parser.add_argument('--val_iteration', default=20, type=int,
                    help='Frequency for validation.')
parser.add_argument('--gpus', default=None, type=str,
                    help='GPUs to use (ex. ''0, 1, 2, 3''). If None, use CPU.')
args = parser.parse_args()


csv_file_train = args.csv_file_train
csv_file_valid = args.csv_file_valid
num_workers = args.num_workers
batch_size = args.batch_size
iterations = args.iterations
lr = args.learning_rate
momentum = args.momentum
val_iter = args.val_iter
val_iteration = args.val_iteration

gpus = args.gpus
use_cuda = False
ngpus = 0
#if gpus is not None:
  #  use_cuda = True
   # ngpus = len(gpus.split(','))

seed = 16
torch.manual_seed(seed)
#if use_cuda:
 #   device = "cuda"
  #  os.environ['CUDA_VISIBLE_DEVICES'] = gpus
   # torch.cuda.manual_seed(seed)

# Transforms data for train
transformed_dataset_train = FallDataset(csv_file=csv_file_train, transform=transforms.Compose([
                                         ToTensor()
                                     ]))

dataloader_train = DataLoader(transformed_dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)

# Transforms data for validation
transformed_dataset_val = FallDataset(csv_file=csv_file_valid, transform=transforms.Compose([
                                         ToTensor()
                                     ]))

dataloader_val = DataLoader(transformed_dataset_val, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)


net = Net()
# Run network on GPU
#if use_cuda:
 #   net = net.cuda()

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer function
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

cur_iter = 0
outfile = open('out_train32_2.txt', "w")

# Train
for i in range(cur_iter, iterations):
    # transform train database
    data = next(iter(dataloader_train))


    # Extract data and target from dictionary
    data, target = \
        data['data'],data['target']

    # Make tensors of train data and target run on GPU
   # if use_cuda:
    #    data = data.cuda()
     #   target = target.cuda()

    #Zeroes the gradient buffers of all parameters
    optimizer.zero_grad()

    # Forword with data in model
    out = net.forward(data)
  

    # Calculate loss for batch size used
    target = target.long()

    loss = criterion(out, target)

    # Backpropogate the error
    loss.backward()

    # Does the update of gradients
    optimizer.step()

    # Print and save loss at each 100 iteration
    if cur_iter % 100 == 0:
        message = 'Iteration: {:.0f}, loss: {:.2f}'.format(cur_iter,loss.data)
        print(message)
        outfile.write(message+'\n')

    # Validation
    if cur_iter % val_iter == 0:
        test_loss = 0
        correct = 0
        total = 0
        # Impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations. Used only for eval.
        with torch.no_grad():            
            net.eval()

            for i in range(val_iteration):
                # transform the val database 
                data = next(iter(dataloader_val))
                # Extract data and target from dictionary
                data, target = \
                data['data'],data['target']
                # Make tensors of val data and target run on GPU
               # if use_cuda:
                #    data = data.cuda()
                 #   target = target.cuda()
                # Forword with val data
                output = net.forward(data)
                # Compute loss
                target = target.long()
                test_loss += criterion(output, target).item() 
                # Predict outputs for each sample
                pred = output.argmax(dim=1, keepdim=True) 
                # Number of correct outputs
                correct += pred.eq(target.view_as(pred)).sum().item()
                # Number of samples used for validation
                total += batch_size

        # Mean loss 
        test_loss /= total

        message_val = 'Validation at iteration {}: loss: {:.4f}, acc: {:.4f}'.format(
                cur_iter, test_loss, correct / total)




        # Write loss and accuracy in txt
        outfile.write(message_val+'\n')
        print(message_val)

        # Path to save checkpoint
        model_name =('log/model_%d.pt'% cur_iter)

        # Save checkpoint
        torch.save({
          'cur_iter': cur_iter,
            'model_state_dict': net.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': lr,
            'loss': loss
        }, model_name)
    cur_iter = cur_iter + 1
net.eval()
exemplu = [0.0,0.1,0.2]
input_tensor = torch.tensor([exemplu])
script_module = torch.jit.trace(net,input_tensor)
script_module.save("model_v7_final.pt")
