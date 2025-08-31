#run.py  you can execute all files in google collab , each file is one cell
import torch.optim as optim

# Initialize model, optimizer, and loss function
print(f'Dataset: {DATASET}, Learning Rate: {LR}\n')

# Initialize the model with the xLSTM-based architecture
model = DKT_xLSTM(INPUT, HIDDEN, LAYERS, OUTPUT).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=LR)

loss_func = lossFunc()

# Get train and test data loaders
train_loaders, test_loaders = getLoader(DATASET)

# Train and evaluate the model
for epoch in range(EPOCH):
    print(f'Epoch: {epoch}')
    for loader in train_loaders:
        model, optimizer = train_epoch(model, loader, optimizer, loss_func)
    for loader in test_loaders:
        test(loader, model)