#  /*******************************************************************************
#   * Copyright 2022 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#
import gc
import torch


class Trainer(object):
    checkpoint_path = '.checkpoint.pt'

    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.losses = []
        self.accuracies = []
        self.epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def restore_from_checkpoint(self):
        checkpoint = torch.load(Trainer.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.__optim_to_device__()
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.losses = checkpoint['losses'].tolist()
        self.accuracies = checkpoint['accuracies'].tolist()
        self.epoch = checkpoint['epoch'] + 1
        print('Restoring after epoch {}.'.format(self.epoch))

    def train(self, epochs):
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

            if self.scheduler is not None:
                self.scheduler.step()
            
            accuracy = self.validate()
            self.losses.append(loss.item())
            self.accuracies.append(accuracy)
            
            print('Epoch [{}], Loss: {:.4f}, Accuracy: {}%'.format(epoch + 1, loss.item(), accuracy), flush=True)
            self.__checkpoint__(epoch)

    def validate(self):
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
            
            return 100 * correct / total
        
    def __checkpoint__(self, epoch):
        torch.save({
            'epoch': epoch,
            'losses': torch.FloatTensor(self.losses),
            'accuracies':torch.FloatTensor(self.accuracies),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': None if self.scheduler is None else self.scheduler.state_dict(),
        }, Trainer.checkpoint_path)
        
    def __optim_to_device__(self):
        if torch.cuda.is_available():
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
