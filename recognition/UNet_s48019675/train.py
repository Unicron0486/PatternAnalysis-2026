from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import modules
import dataset
import utils

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
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

def validate_2d(loader, model, criterion):
    model.eval()
    running_loss = 0.0

    for images, ground_truth in loader:
        images = images.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, ground_truth)

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
    data = dataset.UNet2d_Data()
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = modules.twoD_UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_dice = 0.0

    training_losses = []
    validation_losses = []
    dice_scores = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch_2d(train_loader, model, optimizer, criterion)
        val_loss = validate_2d(val_loader, model, criterion)
        val_dice = evaluate_2d(val_loader, model)

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
                "best_unet.pth"
            )

            print(
                f"Saved model (Dice={val_dice:.4f})"
            )

    make_plots_2d(training_losses, validation_losses, dice_scores)

        

    # dtrain = dataset.get_training_2d()
    # dtrain_loader = torch.utils.data.DataLoader(dtrain, batch_size=BATCH_SIZE, shuffle=True)

    # # Train
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # model = modules.twoD_UNet().to(device)

    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
