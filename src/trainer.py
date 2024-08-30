import torch

def trainer(model, criterion, optimizer, train_loader, valid_loader, device, epochs=10, verbose=True):
    train_accuracy = []
    valid_accuracy = []

    for epoch in range(epochs):
        train_batch_loss = 0
        train_batch_acc = 0
        valid_batch_loss = 0
        valid_batch_acc = 0
        
        # Training loop
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y.long())
            loss.backward()
            optimizer.step()
            train_batch_loss += loss.item()
            _, y_hat_labels = torch.max(y_hat, 1)
            train_batch_acc += (y_hat_labels == y).type(torch.float32).mean().item()
        train_accuracy.append(train_batch_acc / len(train_loader))
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y.long())
                valid_batch_loss += loss.item()
                _, y_hat_labels = torch.max(y_hat, 1)
                valid_batch_acc += (y_hat_labels == y).type(torch.float32).mean().item()
        valid_accuracy.append(valid_batch_acc / len(valid_loader))
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch + 1}: Train Accuracy: {train_accuracy[-1]:.2f} Valid Accuracy: {valid_accuracy[-1]:.2f}")

    return {"train_accuracy": train_accuracy, "valid_accuracy": valid_accuracy}