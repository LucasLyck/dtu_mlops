import argparse
import sys

import matplotlib.pyplot as plt

import torch
import click
import warnings
from torch import nn, optim

from data import mnist
from model import MyAwesomeModel

warnings.simplefilter(action='ignore', category=FutureWarning)

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    # load model
    model = MyAwesomeModel()
    print(model)
    # get train and test dataloader 
    train_dataloader, test_dataloader = mnist()

    images, labels = next(iter(test_dataloader))

    # convert images and labels dtype to same dtype as model.
    images = images.to(next(model.parameters()).dtype)
    labels = labels.to(next(model.parameters()).dtype)

    # accuracy stuff
    ps = torch.exp(model(images))
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))

    # Hyper parametters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    epochs = 5

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_dataloader:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            # validate the model
            model.eval()  # switch to evaluation mode
            with torch.no_grad():
                accuracy = 0
                valid_loss = 0
                for images, labels in test_dataloader:
                    log_ps = model(images)
                    valid_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                valid_loss /= len(test_dataloader)
                accuracy /= len(test_dataloader)
            model.train()  # switch back to training mode
            
            train_losses.append(running_loss/len(train_dataloader))
            test_losses.append(valid_loss)

            print(f'Epoch: {e+1}/{epochs}.. ',
                f'Train loss: {running_loss/len(train_dataloader):.3f}.. ',
                f'Valid loss: {valid_loss:.3f}.. ',
                f'Valid accuracy: {accuracy.item()*100}%')

    torch.save(model.state_dict(), "checkpoint")
    
    plt.plot(train_losses, label="Training loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()







@click.command()
@click.argument("model_checkpoint")




def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # TODO: Implement evaluation logic here

    _, test_dataloader = mnist()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy on test data: {}%".format(100 * correct / total))
    
    # model = torch.load(model_checkpoint)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    