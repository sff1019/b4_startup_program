import time

from evaluator import evaluate_all


class Trainer(object):
    def __init__(self, criterion, device, model, optimizer, test_loader, train_loader, epochs=50):
        self.criterion = criterion
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.train_loader = train_loader

        self.epochs = epochs

        self.train_results = None

    def run(self):
        start_time = time.time()
        results = []

        for epoch in range(self.epochs):
            print(f'{epoch} is running...')
            running_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                if self.device != 'cpu':
                    # Send inputs and labels to GPU
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                # Initialize optimizer
                self.optimizer.zero_grad()

                # Get training result
                outputs = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, labels)

                # Backward propagation
                loss.backward()

                # Updata params
                self.optimizer.step()

                running_loss += loss.item()

                if i % 12500 == 12499:
                    # After every epoch, obtain the results for that epoch
                    epoch_result = {}
                    epoch_result['epoch'] = epoch
                    epoch_result['main/loss'] = float(running_loss / 12500)
                    epoch_result['main/accuracy'] = evaluate_all(
                        self.device,
                        self.train_loader,
                        self.model,
                    )
                    epoch_result['test/accuracy'] = evaluate_all(
                        self.device,
                        self.test_loader,
                        self.model,
                    )
                    epoch_result['elapsed_time'] = time.time() - start_time

                    results.append(epoch_result)
                    running_loss = 0.0

        self.train_results = results
        print('Finished Training')
