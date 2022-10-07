from torch import nn, optim

def create_model():
    # Build a feed-forward network
    input_size = 784 #28x28
    output_size = 10
    model = nn.Sequential(nn.Linear(input_size, 128), #Performs W.x + b
                          nn.ReLU(),                  #Adds Non-Linearity
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, output_size),
                          nn.LogSoftmax(dim=1))

    return model

#NOTE: Do not change any of the variable names to ensure that the training script works properly

model=create_model()

cost = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
