import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def create_training_set():

    towns = open('greek_towns.txt', 'r', encoding='utf-8').read().splitlines()
    
    #Trigram --> 2 characters as input to create a third one
    xs, ys = [], []

    #25 english letters (no 'q') + start/end character + '-' used for replacing spaces = 27
    N = torch.zeros((27,27), dtype=torch.int32)
    chars = sorted(list(set(''.join(towns))))
    #print(chars)

    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    
    for t in towns:
        chs = ['.'] + list(t) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            xs.append([ix1, ix2])
            ys.append(ix3)
        
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

     # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(xs, ys, test_size=0.1, random_state=42)  # 80% train, 20% temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split temp into 10% val, 10% test

     # One-hot encode the inputs
    X_train_enc = F.one_hot(X_train, num_classes=27).float()
    X_val_enc = F.one_hot(X_val, num_classes=27).float()
    X_test_enc = F.one_hot(X_test, num_classes=27).float()

    #Create one-hot encodings of characters
    xenc = F.one_hot(xs, num_classes=27).float()
    #print(xenc)
    #print(xenc.shape) # 45971 pairs of 2 characters encoded by 27x1 one-hot encodings
    #print(xenc.dtype)
    #yenc = F.one_hot(ys, num_classes=27).float()

    return X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test, stoi, itos

class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        
        self.fc1 = nn.Linear(54, 64)  # First hidden layer: 54 = 2x27
        self.fc2 = nn.Linear(64, 32)  # Second hidden layer 
        self.fc3 = nn.Linear(32, 27)  # Output layer with 27 neurons
    
        self.activation = nn.Tanh()
        #self.log_softmax = nn.LogSoftmax(dim=1) #For negative log likelihood loss

        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Flatten input from (batch_size, 2, 27) to (batch_size, 54)
        x = x.view(x.size(0), -1)
        
        # Pass through the layers
        x = self.activation(self.fc1(x))
        #x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation here, raw logits returned
        #x = self.log_softmax(x)  # Convert logits to log probabilities (for negative log-likelihood loss)
        
        return x
    
def model_training(X_train, y_train, X_val, y_val):

    model = FullyConnectedNN().to('cuda')
    # Loss function: CrossEntropyLoss expects raw logits (so no softmax in the model)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    num_epochs = 100
    best_val_loss = float('inf')

    print("\nTraining starting...")
    for epoch in range(num_epochs):
        model.train()
        # Zero the gradients
        optimizer.zero_grad()

        output = model(X_train)
        #print(output.shape) #Samples X 27

        # Compute loss
        train_loss = criterion(output, y_train)

        # Backward pass
        train_loss.backward()

        # Update the weights
        optimizer.step()

        # Step the learning rate scheduler
        #scheduler.step()

         # Validation step (set model to eval mode)
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_trigram_nn.pt")

    print("Training ended!\n")

    #torch.save(model.state_dict(), "models/trigram_nn.pt")

    return model

def get_first_char_samples(samples):

    towns = open('greek_towns.txt', 'r', encoding='utf-8').read().splitlines()
    first_char_distribution = {}

    for t in towns:
        if t[0] in first_char_distribution:
            first_char_distribution[t[0]] += 1.0
        else:
            first_char_distribution[t[0]] = 0.0

    #Transform to probability vector
    all_first_chars = []
    fisrt_chars_probs = []
    for letter in first_char_distribution:
        first_char_distribution[letter] = first_char_distribution[letter]/float(len(towns))
        all_first_chars.append(letter)
        fisrt_chars_probs.append(first_char_distribution[letter])

    g = torch.Generator().manual_seed(2147483647)
    sample_chars = torch.multinomial(torch.tensor(fisrt_chars_probs), num_samples=samples, replacement=True, generator=g)

    first_chars = []
    for i in sample_chars:
        first_chars.append(all_first_chars[i.item()])
    
    return first_chars

def generate_cities(model, num_cities, first_chars_sample, stoi, itos):

    #We have to get a random intial character first since our model in a trigram
    # It needs 2 characters as input: The start character and a first letter
    # We will count the frequency of all letters as first letters 
    # Alternative: Just picking a random character as a first character.

    # print(stoi)
    # print(itos)
    print("\nInference: Generating city names")

    for i in range(num_cities):

        out = [first_chars_sample[i]]
        ix1 = 0 # A word always start with a start character
        ix2 = stoi[first_chars_sample[i]]

        while True:
            xenc = F.one_hot(torch.tensor([[ix1, ix2]]), num_classes=27).float().to('cuda')
            with torch.no_grad():  # Disable gradient calculation
                model_output = model(xenc)
            ix1 = ix2
            
            #print(model_output)
            # Convert logits to probabilities 
            probabilities = torch.softmax(model_output, dim=1)
            
            # Get the predicted class
            # Different alternatives here: 
            # Either stohastically picking a class or picking the best probability
            # The best probability usually returns repeatable results
            predicted_class = torch.multinomial(probabilities, 1)
            #_, predicted_class = torch.max(probabilities, 1)

            predicted_char = itos[predicted_class.item()]

            out.append(predicted_char)
            ix2 = stoi[predicted_char]

            if ix2 == 0:
                break
        
        print(''.join(out[:-1]))
    return

def evaluate_on_test_set(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = nn.CrossEntropyLoss()(output, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')

if __name__ == "__main__":

    print("Cities trigram language model!")
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    X_train, y_train, X_val, y_val, X_test, y_test, stoi, itos = create_training_set()
    model = model_training(X_train.to('cuda'), y_train.to('cuda'), X_val.to('cuda'), y_val.to('cuda'))
    evaluate_on_test_set(model, X_test.to('cuda'), y_test.to('cuda'))
    
    #Load model
    # model = FullyConnectedNN()
    # model.load_state_dict(torch.load("models/trigram_nn.pt"))
    # model.to('cuda')
    model.eval() #For inference

    cities_to_generate = 30
    first_chars_sample = get_first_char_samples(cities_to_generate)
    generate_cities(model, cities_to_generate, first_chars_sample, stoi, itos)
    
    
        


