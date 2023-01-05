"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt
import wandb

dataset_path = 'datasets'
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
#batch_size = 100
x_dim  = 784
hidden_dim = 400
latent_dim = 20
#learning_rate = 1e-3
#epochs = 5

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')

def compute_validation_metrics(model, dataloader):
    ''' The prettiest function 🙂'''
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
    return total_loss / len(dataloader), total_acc / len(dataloader)

# from torch.profiler import profile, tensorboard_trace_handler
# with profile(..., on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:

def main():
    run = wandb.init()

    learning_rate  =  wandb.config.lr
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

    class Encoder(nn.Module):  
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(Encoder, self).__init__()
            
            self.FC_input = nn.Linear(input_dim, hidden_dim)
            self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
            self.FC_var   = nn.Linear (hidden_dim, latent_dim)
            self.training = True
            
        def forward(self, x):
            h_       = torch.relu(self.FC_input(x))
            mean     = self.FC_mean(h_)
            log_var  = self.FC_var(h_)                     
                                                        
            std      = torch.exp(0.5*log_var)             
            z        = self.reparameterization(mean, std)
            
            return z, mean, log_var
        
        def reparameterization(self, mean, std,):
            epsilon = torch.randn_like(std)
            
            z = mean + std*epsilon
            
            return z
        
    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim, output_dim):
            super(Decoder, self).__init__()
            self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
            self.FC_output = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            h     = torch.relu(self.FC_hidden(x))
            x_hat = torch.sigmoid(self.FC_output(h))
            return x_hat
        
    class Model(nn.Module):
        def __init__(self, Encoder, Decoder):
            super(Model, self).__init__()
            self.Encoder = Encoder
            self.Decoder = Decoder
                    
        def forward(self, x):
            z, mean, log_var = self.Encoder(x)
            x_hat            = self.Decoder(z)
            
            return x_hat, mean, log_var
        
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    from torch.optim import Adam

    BCE_loss = nn.BCELoss()

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    optimizer = Adam(model.parameters(), lr=learning_rate)

    
    print("Start training VAE...")
    model.train()
    train_loss = []
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss) 
        
        train_loss, train_acc = compute_validation_metrics(model,train_loader)
        val_loss, val_acc = compute_validation_metrics(model,test_loader)

        wandb.log({
            'epoch': epoch, 
            'train_acc': train_acc,
            'train_loss': train_loss, 
            'val_acc': val_acc, 
            'val_loss': val_loss
        })

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))    
    print("Finish!!")

wandb.agent(sweep_id, function=main, count=4)
# Generate reconstructions
# model.eval()
# with torch.no_grad():
#     for batch_idx, (x, _) in enumerate(test_loader):
#         x = x.view(batch_size, x_dim)
#         x = x.to(DEVICE)      
#         x_hat, _, _ = model(x)       
#         break

# save_image(x.view(batch_size, 1, 28, 28), 'orig_data.png')
# save_image(x_hat.view(batch_size, 1, 28, 28), 'reconstructions.png')

# # Generate samples
# with torch.no_grad():
#     noise = torch.randn(batch_size, latent_dim).to(DEVICE)
#     generated_images = decoder(noise)
    

# save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')
# wandb.log({"generated_images" : wandb.Image(generated_images)})
# wandb.log({"origin_data" : wandb.Image(x)})
# wandb.log({"reconstructions" : wandb.Image(x_hat)})


## 2 - Adam

## 3 - tottime - total time used by the function
##     cumtime - total time used by the function and the ones it calls