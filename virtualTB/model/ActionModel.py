import os
from virtualTB.utils import *

class ActionModel(nn.Module):
    def __init__(self, n_input = 88 + 1 + 27, n_output = 11 + 10, learning_rate = 0.01):
        super(ActionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)
        )
        self.max_a = 11
        self.max_b = 10

    def predict(self, user, page, weight):
        #x 88+1+27d input, 21d output
        x = self.model(torch.cat((user, page, weight), dim = -1))
        
        #在前11维中采样最大的值，在后10维中采样最大的值，然后拼接
        a = torch.multinomial(F.softmax(x[:, :self.max_a], dim = 1), 1)
        b = torch.multinomial(F.softmax(x[:, self.max_a:], dim = 1), 1)
        
        # 2d
        return torch.cat((a, b), dim = -1)

    def load(self, path = None):
        if path == None:
            path = os.path.dirname(__file__) + '/../data/action_model.pt'
        self.model.load_state_dict(torch.load(path))