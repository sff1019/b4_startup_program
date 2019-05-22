import time

from evaluator import evaluate_all
from log import plot_log


def train(
        device,
        epochs,
        loss_fn,
        model,
        optimizer,
        start_time,
        test_loader,
        train_loader,
):
    """
    Train data and return results
    return: epoch, main/loss, main/accuracy, test/accuracy and elapsed_time
    """
    results = []
    for epoch in range(epochs):
        print(f'{epoch} is running...')
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Initialize optimizers
            optimizer.zero_grad()

            outputs = model(inputs)

            # Calculate the loss
            loss = loss_fn(outputs, labels)

            # Backward propagation
            loss.backward()

            # Update params
            optimizer.step()

            running_loss += loss.item()

            if i % 12500 == 12499:
                # After every epoch, obtain the results for that epoch
                epoch_result = {}
                epoch_result['epoch'] = epoch
                epoch_result['main/loss'] = float(running_loss / 12500)
                epoch_result['main/accuracy'] = evaluate_all(
                    train_loader, model, device)
                epoch_result['test/accuracy'] = evaluate_all(
                    test_loader, model, device)
                epoch_result['elapsed_time'] = time.time() - start_time

                results.append(epoch_result)

                # Initialize the loss
                running_loss = 0.0

    print('Finished Training')

    return results
