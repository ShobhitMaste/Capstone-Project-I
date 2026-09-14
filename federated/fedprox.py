import torch

class FedProxLoss(torch.nn.Module):
    def __init__(self, mu=0.01):
        super(FedProxLoss, self).__init__()
        self.mu = mu
        self.base_loss = torch.nn.CrossEntropyLoss()
        
    def forward(self, outputs, labels, model, global_params):
        # Base loss
        loss = self.base_loss(outputs, labels)
        
        # Proximal term
        proximal_term = 0.0
        
        for param, global_param in zip(model.parameters(), global_params):
            proximal_term += torch.square(torch.norm(param - global_param))
            
        return loss + (self.mu / 2) * proximal_term

def train_fedprox(model, train_loader, global_params, epochs, lr, mu, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = FedProxLoss(mu=mu)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # FedProx loss requires global_params
            loss = criterion(outputs, labels, model, global_params)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    avg_loss = total_loss / max(1, total)
    avg_acc = correct / max(1, total)
    
    return avg_loss, avg_acc
