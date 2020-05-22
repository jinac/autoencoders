"""
util
"""

def save_weights(model, filepath):
    torch.save(model.state_dict(), filepath)