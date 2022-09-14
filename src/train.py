import gc
import torch


class Trainer(object):
    checkpoint_path = '.checkpoint.pt'

    def __init__(self, model, train_loader, valid_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.losses = []
        self.epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def restore_from_checkpoint(self):
        checkpoint = torch.load(Trainer.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint['losses'].tolist()
        self.epoch = checkpoint['epoch'] + 1
        print('Restoring after epoch {}.'.format(self.epoch))

    def train(self, epochs, save_freq: int=100):
        # Set to training mode
        self.model.train()
        self.criterion.train()

        # run on GPU if available
        self.model.to(self.device)

        # Training loop
        for epoch in range(self.epoch, epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del images, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()
            
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
            self.losses.append(loss.item())

            if epoch % save_freq == 0:
                self.__checkpoint__(epoch)

        self.__checkpoint__(epochs-1)
        self.__validate__()

        return self.losses

    def __validate__(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.valid_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy for the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

    def __checkpoint__(self, epoch):
        torch.save({
            'epoch': epoch,
            'losses': torch.FloatTensor(self.losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, Trainer.checkpoint_path)
