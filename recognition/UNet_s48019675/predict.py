

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
    torch.cuda.reset_peak_memory_stats()
    tr.training_2d()
    if torch.cuda.is_available():
            max_mem = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )
    
    with open("training_log.txt", "a") as log_file:
        log_file.write(
            f"Peak GPU Memory: {max_mem:.2f} GB\n"
        )
    print("testing")
    # model = modules.TwoD_UNet()
    # model.load_state_dict(torch.load("best_unet.pth", map_location=DEVICE))
    # model.eval()
    tr.test_2d()

    print("3d training")
    torch.cuda.reset_peak_memory_stats()
    m = tr.model_3d()
    m.train_model()
    if torch.cuda.is_available():
            max_mem = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )
    
    with open("training_log3d.txt", "a") as log_file:
        log_file.write(
            f"Peak GPU Memory: {max_mem:.2f} GB\n"
        )
    print("3d testing")
    m.test_model()
    # train_3d = tr.training_3d()
    # tr.test_3d()
