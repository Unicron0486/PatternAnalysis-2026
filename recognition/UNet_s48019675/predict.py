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

    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1).float()

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

    with torch.no_grad():

        for image, ground_truth in test_loader:

            image = image.to(DEVICE)
            ground_truth = ground_truth.to(DEVICE)

            output = model(image)

            prediction = torch.softmax(output, dim=1)

            prediction = (
                prediction > 0.5
            ).float()

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

                plt.figure(figsize=(12,4))

                plt.subplot(1,3,1)
                plt.imshow(
                    image[0,0].cpu().numpy(),
                    cmap="gray"
                )
                plt.title("Input")
                plt.axis("off")

                plt.subplot(1,3,2)
                plt.imshow(
                    ground_truth[0,0].cpu().numpy(),
                    cmap="gray"
                )
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1,3,3)
                plt.imshow(
                    prediction[0,0].cpu().numpy(),
                    cmap="gray"
                )
                plt.title("Prediction")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

    average_dice = total_dice / count

    print(
        f"Average Dice Score: {average_dice:.4f}"
    )

if __name__ == "__main__":
    predict_2d()