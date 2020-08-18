import os
import sklearn
import torch
import kornia
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from tqdm import tqdm, trange
from scipy.stats import logistic
from pytorch_transformers.optimization import WarmupLinearSchedule
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse

# Internal Libraries
from parsers import perturb_parser as parser
from models.mit_model import resnet14_1, resnet14_16, resnet14_4, resnet10_16
from models.resnet import ResidualNet
#from dataset import CXRImageDataset_InMem as CXRImageDataset
from peter_dataset import CXRDataset
from test_h5 import evaluate

device = torch.device("cuda")

# Read arguments
args = parser.get_args()
print(args)

''' Verify weights directory exists, if not create it '''
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

all_results = {}
FOLDS = args.training_folds + args.validation_folds
for validation_fold in FOLDS:
    print("Training with fold {} held out for validation".format(validation_fold))
    if not os.path.isdir(args.output_dir + "/" + str(validation_fold)):
        os.makedirs(args.output_dir + "/" + str(validation_fold))

    training_folds = [f for f in FOLDS if f != validation_fold] 

    # Datasets
    train_dataset = CXRDataset(args.image_dir, training_folds)
    val_dataset = CXRDataset(args.image_dir, [validation_fold])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=True)

    print('Total number of training images: {}, validation images: {}.'.format(len(train_dataset), len(val_dataset)))

    aux_criterion = CrossEntropyLoss().to(device) 
    if args.label_encoding == 'ordinal':
        BCE_loss_criterion = BCEWithLogitsLoss().to(device)
        add_softmax = False
        output_channels = 3
    if args.label_encoding == 'onehot':
        BCE_loss_criterion = BCELoss().to(device)
        add_softmax = True
        output_channels = 4

    # Create instance of a resnet model
    if args.model_architecture == 'resnet18':
        model = ResidualNet('ImageNet', 18, output_channels, None, add_softmax=True)
    if args.model_architecture == 'resnet14_1':
        model = resnet14_1(add_softmax=add_softmax, 
                                              output_channels=output_channels, latent_dim=args.latent_dim)
    if args.model_architecture == 'resnet14_16':
        model = resnet14_16(add_softmax=add_softmax, 
                                               output_channels=output_channels, latent_dim=args.latent_dim)
    if args.model_architecture == 'resnet14_4':
        model = resnet14_4(add_softmax=add_softmax, 
                                              output_channels=output_channels, latent_dim=args.latent_dim)
    if args.model_architecture == 'resnet10_16':
        model = resnet10_16(add_softmax=add_softmax, 
                                               output_channels=output_channels, latent_dim=args.latent_dim)
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if 'rot_pred' in name or 'fc' in name:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        return own_state

    model = model.to(device)
    model = nn.DataParallel(model)
    pretrained = torch.load(args.pretrained_dir + str(validation_fold) + "/best.ckpt")
    model.load_state_dict(load_my_state_dict(model, pretrained))

    for name, param in model.named_parameters():
        if 'module.conv1' in name or 'module.bn1' in name or 'layer1' in name or 'layer2' in name or 'layer3' in name:
            param.requires_grad = False

    # Create instance of optimizer (AdamW) and learning rate scheduler 
    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr)

    # Select options per parser args
    if args.scheduler == 'WarmupLinearSchedule':
        num_train_optimization_steps = len(train_loader) * args.num_train_epochs
        args.warmup_steps = args.warmup_proportion * num_train_optimization_steps
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-6)

    transform = nn.Sequential(
        kornia.augmentation.RandomAffine((-20., 20.),translate=(0.1, 0.1))
    )

    # Start model training
    best_loss = np.inf
    total_step = len(train_loader)
    early_stop = 0
    running_loss = 0.0
    for epoch in range(args.num_train_epochs):
        tr_loss = 0
        model.train()
        for i, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            inputs, labels, img_id = batch
            inputs = transform(inputs)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs * 2 - 1, aux=False)
            loss = BCE_loss_criterion(outputs[0], labels)
            # zero the parameter gradients
            '''
            optimizer.step()
            optimizer.zero_grad()


            curr_batch_size = inputs.size(0)
            aux_labels = torch.cat((torch.zeros(curr_batch_size), torch.ones(curr_batch_size),
                                  2*torch.ones(curr_batch_size), 3*torch.ones(curr_batch_size)), 0).long()
            inputs = inputs.cpu().numpy()
            inputs = np.concatenate((inputs, np.rot90(inputs, 1, axes=(2, 3)),
                                 np.rot90(inputs, 2, axes=(2, 3)), np.rot90(inputs, 3, axes=(2, 3))), 0)
            inputs = torch.FloatTensor(inputs)
            inputs, aux_labels = inputs.cuda(), aux_labels.cuda()
            outputs = model(inputs * 2 - 1, aux=True)
            loss += 0.5 * aux_criterion(outputs[0], aux_labels) 
            '''

            loss.backward()
            optimizer.step()

            # Update learning rate schedule
            if args.scheduler == 'WarmupLinearSchedule':
                scheduler.step()

            # print statistics
            running_loss += loss.item()
            tr_loss += loss.item()

            print ("Epoch [{}/{}], Step [{}/{}] Total Loss: {:.4f}, Epoch Loss: {:.4f}, Batch Loss: {:.4f}"\
                    .format(epoch+1, args.num_train_epochs, i+1, total_step, running_loss/((epoch * total_step * args.batch_size) + ((i+1) * args.batch_size)),\
                    tr_loss/((i+1) * args.batch_size), loss.item()/args.batch_size))

        if args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(tr_loss)

        early_stop += 1

        vloss = 0
        model.eval()
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                inputs, labels, img_id = batch

                # Forward pass
                outputs = model(inputs * 2 - 1, aux=False)
                vloss += BCE_loss_criterion(outputs[0], labels)

        epoch_loss = vloss/(len(val_loader) * args.batch_size)
        print("Validation Loss: {:.4f}".format(epoch_loss))
            
        if epoch_loss < best_loss:
            early_stop = 0
            best_loss = epoch_loss
            torch.save(model.state_dict(), args.output_dir + "/" + str(validation_fold) + "/best.ckpt")

        if early_stop == args.early_stopping:
            print("Validation loss has not improved - early stopping!")
            break

    results = evaluate(args, device, model)
    results = results[0]

    if all_results == {}:
        for key, val in results.items():
            all_results[key] = [val]
    else:
        for key, val in results.items():
            all_results[key].append(val)


for key, val in all_results.items():
    print("{} mean: {}, standard deviation: {}".format(key, np.mean(val), np.std(val)))
