"""
util
"""
import torch

def fix_seed():
	# Fix seed.
	torch.manual_seed(1337)
	torch.cuda.manual_seed(1337)
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.deterministic = True

def save_weights(model, filepath):
    torch.save(model.state_dict(), filepath)