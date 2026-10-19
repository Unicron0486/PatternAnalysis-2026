from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import modules
import dataset
import utils
from matplotlib.patches import Patch

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Training validation testing and saving
def train_epoch_2d(loader, model, optimiser, criterion):
    model.train()
    running_loss = 0.0
    for images, ground_truth in loader:
        images = images.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)


        
        outputs = model(images)
        loss = criterion(outputs, ground_truth)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    return epoch_loss

def dice_2d(preds , ground_truth):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * ground_truth).sum()

    dice_score = (2. * intersection + 1e-8) / (preds.sum() + ground_truth.sum() + 1e-8)
    return dice_score.item()

def evaluate_2d(loader, model):
    model.eval()
    dice_total = 0.0
    count = 0
    with torch.no_grad():
        for images, ground_truth in loader:
            images = images.to(DEVICE)
            ground_truth = ground_truth.to(DEVICE)

            predictions = model(images)
            dice_score = dice_2d(predictions, ground_truth)
            dice_total += dice_score
            count += 1

    if count > 0:
        return dice_total / count
    else:
        return 0.0

def make_plots_2d(training_losses, validation_losses, dice_scores):
    plt.figure()
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend([("Training Loss"), ("Validation Loss")])

    plt.savefig("loss_plot.png")

    plt.figure()
    plt.plot(dice_scores, label="Dice Score")
    plt.savefig("dice_plot.png")


def training_2d():
    # Load Data
    train_dataset = dataset.get_training_2d()
    val_dataset = dataset.get_validation_2d()
    print("data loaded")

    # test_dataset = dataset.get_testing_2d()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = modules.TwoD_UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    best_dice = 0.0

    training_losses = []
    validation_losses = []
    dice_scores = []
    with open("training_log.txt", "w") as log_file:
        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch_2d(train_loader, model, optimizer, criterion)
            val_loss = validate_2d(val_loader, model, criterion)
            val_dice = evaluate_2d(val_loader, model)

            training_losses.append(train_loss)
            validation_losses.append(val_loss)
            dice_scores.append(val_dice)

            log_file.write(
                f"Epoch {epoch+1:03d} "
                f"Train Loss {train_loss:.4f} "
                f"Val Loss {val_loss:.4f} "
                f"Dice {val_dice:.4f}\n"
            )

            if val_dice > best_dice:

                best_dice = val_dice

                torch.save(
                    model.state_dict(),
                    "best_unet.pth"
                )

                log_file.write(
                    f"Saved model (Dice={val_dice:.4f})\n"
                )
            print("epoch done")
        log_file.flush()  # optional: writes immediately to disk
                
    log_file.close()
    print("training done")
    make_plots_2d(training_losses, validation_losses, dice_scores)

        

    # dtrain = dataset.get_training_2d()
    # dtrain_loader = torch.utils.data.DataLoader(dtrain, batch_size=BATCH_SIZE, shuffle=True)

    # # Train
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # model = modules.twoD_UNet().to(device)

    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def validate_2d(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, ground_truth in loader:
            images = images.to(DEVICE)
            ground_truth = ground_truth.to(DEVICE)

            
            outputs = model(images)
            loss = criterion(outputs, ground_truth)

            running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    return epoch_loss

def test_2d():
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

                plt.savefig(
                        f"prediction_{count}.png",
                        dpi=300,
                        bbox_inches="tight"
                    )

                plt.close()

    average_dice = total_dice / count

    with open("training_log.txt", "a") as log_file:
        log_file.write(
            f"Average Dice Score: {average_dice:.4f}\n"
        )
    log_file.close()




# 3D model
def train_epoch_3d(loader, model, optimiser, criterion):
    model.train()
    running_loss = 0.0
    for images, ground_truth in loader:
        images = images.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, ground_truth)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    return epoch_loss

def validate_3d(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, ground_truth in loader:
            images = images.to(DEVICE)
            ground_truth = ground_truth.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, ground_truth)

            running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    return epoch_loss

def dice_3d(preds , ground_truth):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * ground_truth).sum()

    dice_score = (2. * intersection + 1e-8) / (preds.sum() + ground_truth.sum() + 1e-8)
    return dice_score.item()

def evaluate_3d(loader, model):
    model.eval()
    dice_total = 0.0
    count = 0
    for images, ground_truth in loader:
        images = images.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)

        predictions = model(images)
        dice_score = dice_3d(predictions, ground_truth)
        dice_total += dice_score
        count += 1

    if count > 0:
        return dice_total / count
    else:
        return 0.0

def make_plots_3d(training_losses, validation_losses, dice_scores):
    plt.figure()
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend([("Training Loss"), ("Validation Loss")])

    plt.savefig("loss_plot_3d.png")

    plt.figure()
    plt.plot(dice_scores, label="Dice Score")
    plt.savefig("dice_plot_3d.png")

def training_3d():
    # Load Data
    train_dataset = dataset.get_training_3d()
    val_dataset = dataset.get_validation_3d()
    
    # tester = train_dataset[0]
    # valy = val_dataset[0]
    # print(tester.shape)
    # print(valy.shape)

    # test_dataset = dataset.get_testing_3d()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = modules.ThreeD_UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    
    best_dice = 0.0

    training_losses = []
    validation_losses = []
    dice_scores = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch_3d(train_loader, model, optimizer, criterion)
        val_loss = validate_3d(val_loader, model, criterion)
        val_dice = evaluate_3d(val_loader, model)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        dice_scores.append(val_dice)

        print(
            f"Epoch {epoch+1:03d} "
            f"Train Loss {train_loss:.4f} "
            f"Val Loss {val_loss:.4f} "
            f"Dice {val_dice:.4f}"
        )

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                model.state_dict(),
                "best_unet_3d.pth"
            )

            print(
                f"Saved model (Dice={val_dice:.4f})"
            )

    make_plots_3d(training_losses, validation_losses, dice_scores)



