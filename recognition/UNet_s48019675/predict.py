from matplotlib.patches import Patch

import train
import torch
import matplotlib.pyplot as plt
import modules
import dataset

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def dice_2d(preds, ground_truth):

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * ground_truth).sum()

    dice_score = (
        2.0 * intersection + 1e-8
    ) / (
        preds.sum()
        + ground_truth.sum()
        + 1e-8
    )

    return dice_score.item()


def predict_2d():

    test_data = dataset.get_testing_2d()


    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False
    )

    model = modules.TwoD_UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "best_unet.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    total_dice = 0.0
    count = 0

    print(ground_truth.shape)
    
    with torch.no_grad():

        for image, ground_truth in test_loader:

            image = image.to(DEVICE)
            ground_truth = ground_truth.to(DEVICE)

            output = model(image)

            prediction = torch.sigmoid(output)

            prediction = (
                prediction > 0.5
            ).float()

            combined = torch.argmax(prediction, dim=1)
            combined_truth = torch.argmax(ground_truth, dim=1)

            dice = dice_2d(
                output,
                ground_truth
            )

            total_dice += dice

            count += 1

            #
            # Display first few predictions
            #

            if count <= 5:

                plt.figure(figsize=(16,4))

                plt.subplot(1,4,1)
                plt.imshow(
                    image[0,0].cpu().numpy(),
                    cmap="gray"
                )
                plt.title("MRI")
                plt.axis("off")

                plt.subplot(1,4,2)
                plt.imshow(
                    combined_truth[0].cpu().numpy(),
                    cmap="tab10",
                    vmin=0,
                    vmax=4
                )
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1,4,3)
                plt.imshow(
                    combined[0].cpu().numpy(),
                    cmap="tab10",
                    vmin=0,
                    vmax=4
                )
                plt.title("Prediction")
                plt.axis("off")

                plt.subplot(1,4,4)
                plt.imshow(
                    image[0,0].cpu().numpy(),
                    cmap="gray"
                )

                plt.imshow(
                    combined[0].cpu().numpy(),
                    cmap="tab10",
                    alpha=0.4,
                    vmin=0,
                    vmax=4
                )

                plt.title("Overlay")
                plt.axis("off")

                legend_items = [
                    Patch(color=plt.cm.tab10(0), label="Body"),
                    Patch(color=plt.cm.tab10(1), label="Bone"),
                    Patch(color=plt.cm.tab10(2), label="Bladder"),
                    Patch(color=plt.cm.tab10(3), label="Rectum"),
                    Patch(color=plt.cm.tab10(4), label="Prostate")
                ]

                plt.legend(
                    handles=legend_items,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left"
                )

                plt.tight_layout()
                plt.show()

    average_dice = total_dice / count

    print(
        f"Average Dice Score: {average_dice:.4f}"
    )

if __name__ == "__main__":
    predict_2d()