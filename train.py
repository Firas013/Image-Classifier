# Imports, the same of the workspace 1 ... except argparse
import torch
from torchvision import transforms, datasets
from torchvision import models
from collections import OrderedDict
from torch import optim
from torch import nn
from torch.nn import functional as F
import os
import numpy as np

# New import, which is facilitates deal with command-line interface.
import argparse


# Step 1 : argparse 

''' Demonstrating some point for me
dest = attirbute name
action = what action should be taken when the argument is encountered on the command line. (store), (count), (store_true)
type = the type of arguemnt which might be int, float, str.....
default = providing sensible default values for optional arguments 
'''


# Notice : I used the same codes that I used it in my worspace 1

def train_parser():
    parser = argparse.ArgumentParser(description="Neural network (train) - Image classifier workspace 2")
    
    # directory to the file and the saving
    parser.add_argument('data_dir', action='store', help='The directory to flower training data')
    
    parser.add_argument('--save_dir', action='store', help='The directory of saving the model')
    # Architecture for the model, I preferred to use the same one I used it in workspace 1 ( vgg11 ) 
    parser.add_argument('--arch', action='store',dest='arch',default='vgg11',help='Architecture of the model')
    
    # Learning rate... with assigning the default 0.01  
    parser.add_argument('learning_rate', type=float, default=0.01, help='The learning rate for the model')
    
    # Hidding layers for the model, of course int... 
    parser.add_argument('--hidden_units', type=int, help='Hidden units for (model classifier)')
    
    # Epochs, also int...
    parser.add_argument('--epochs', type=int, default=3 ,help='Number of epochs for training')
    
    # GPU
    parser.add_argument('--gpu', action="store_true", default=False, help='Enable GPU and CUDA')
    
    args = parser.parse_args()
    return args





def model_classifier(model, hidden_units, dropout):
    model = models.vgg11(pretrained=True)
    model.name = "vgg11"
    input_units = model.classifier[0].in_features
    for param in model.parameters():
        param.requires_grad = False 
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    return model 




criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)




def model_training(model, epochs, criterion, optimizer):
    print_for_batches = 32
    epochs = 3
    batch_step = 0
    training_loss = 0
    for epoch in range(epochs):
    
    # Important to set the model in train mode
    model.train()
    
    
        for e, (inputs, labels) in enumerate(train_Data):
        
        
            batch_step += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
        
            optimizer.zero_grad()
        
        
            output = model(inputs)
        
        
            loss = criterion(output, labels)
        
        
            loss.backward()
            optimizer.step()
        
            training_loss = loss.item()
        
        
            if batch_step % print_for_batches == 0:
            
            
                model.eval()
           
                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0
            
                
                    for e, (v_inputs, v_labels) in enumerate(valid_Data):
                        v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                
                        v_output = model(v_inputs)
                        valid_loss = criterion(v_output, v_labels).item()
                    
                        torch_exp = torch.exp(v_output)
                        top_p, top_class = torch_exp.topk(1, dim=1)
                        equals = top_class == v_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                
                print('Epoch {} / {} '.format(epoch+1, epochs),
                  '| Training Loss {:.3f}'.format(training_loss / print_for_batches),
                  '| Validation Loss {:.3f}'.format(valid_loss / len(valid_Data)),
                  '| Validation Accuracy {:.3f}'.format(100 * (accuracy / len(valid_Data))))
            
                training_loss = 0
                model.train() 
            

            
            
def model_accuracy(model):
    correct_prediction = 0
    total_images = 0
    model.to('cuda')
    model.eval()

    with torch.no_grad():
        for ii,(inputs, labels) in enumerate(test_Data):
            inputs, labels = inputs.to(device), labels.to(device)
        
            output_prediction =  model(inputs)
        
        
            _,predicited = torch.max(output_prediction, 1)
        
        
            total_images += labels.size(0)
        
            correct_prediction += (predicited == labels).sum().item()
        
    print('The accuracy that the model has achieved it is : %d%%' % (100 * correct_prediction / total_images))
    
    


    
    
    
def save_model(model,save_dir):
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'epochs' : epochs,
             'state_dict': model.state_dict(),
              'map' : model.class_to_idx,
             'classifier' : model.classifier}

    torch.save(checkpoint, save_dir)
    
    
    return model



def main():
    # Parse command-line arguments
    args = train_parser()

    # Check if GPU is available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load data
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define data transformations
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Load the pretrained model
    model = models.vgg11(pretrained=True)

    # Build the classifier
    model = model_classifier(model, args.hidden_units, 0.2)  # You can adjust dropout rate as needed

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Move model to the device (GPU or CPU)
    model.to(device)

    # Train the model
    model_training(model, args.epochs, criterion, optimizer, train_loader, valid_loader, device)

    # Test the model accuracy
    model_accuracy(model, test_loader, device)

    # Save the trained model
    if args.save_dir:
        save_model(model, args.save_dir)

if __name__ == "__main__":
    main()
