import torch


def evaluate_all(loader, model):
    """
    Evaluate the accuracy of all images
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    result = 100 * correct / total

    return result


def evaluate_classes(loader, model, classes):
    """
    Evaluate the accuracy based on classes
    """
    results = []
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        results.append({classes[i]: 100 * class_correct[i] / class_total[i]})

    return results
