

import train as tr
import torch
import matplotlib.pyplot as plt
import modules
import dataset

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# model = modules.TwoD_UNet()
# model.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
# model.eval()



if __name__ == "__main__":
    tr.training_2d()
    print("testing")
    tr.test_2d()
