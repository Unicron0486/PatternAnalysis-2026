

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
    # tr.training_2d()
    # print("testing")
    # model = modules.TwoD_UNet()
    # model.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
    # model.eval()
    # tr.test_2d()

    m = tr.model_3d()
    m.train_model()
    m.test_model()
    # train_3d = tr.training_3d()
    # tr.test_3d()
