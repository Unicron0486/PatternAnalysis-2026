import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import modules
import dataset
import utils
from matplotlib.patches import Patch

BATCH_SIZE = 35
LEARNING_RATE = 0.001
NUM_EPOCHS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_per_class(preds, ground_truth, num_classes=6):
    """
    Calculate the Dice coefficient for each class in a multi-class segmentation task.
    """

    preds = torch.argmax(preds, dim=1)

    scores = {}

    for c in range(num_classes):

        pred_c = (preds == c).float()
        gt_c = (ground_truth == c).float()

        if gt_c.sum() == 0:
            continue

        intersection = (pred_c * gt_c).sum()

        dice = (
            2.0 * intersection + 1e-8
        ) / (
            pred_c.sum()
            + gt_c.sum()
            + 1e-8
        )

        scores[c] = dice.item()

    return scores

def iou_per_class(preds, ground_truth, num_classes=6):
    """
    Calculate the Intersection over Union (IoU) for each class in a multi-class segmentation task.
    """

    preds = torch.argmax(preds, dim=1)

    scores = {}

    for c in range(num_classes):

        pred_c = (preds == c)
        gt_c = (ground_truth == c)

        if gt_c.sum() == 0:
            continue

        intersection = (pred_c & gt_c).sum()

        union = (pred_c | gt_c).sum()

        iou = (
            intersection + 1e-8
        ) / (
            union + 1e-8
        )

        scores[c] = iou.item()

    return scores

#Training validation testing and saving
def train_epoch_2d(loader, model, optimiser, criterion):
    """
    Train the model for one epoch on 2D data.
    """
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

def dice_2d(preds, ground_truth, num_classes=6):
    """
    Calculate the Dice coefficient for 2D segmentation.
    """
    preds = torch.argmax(preds, dim=1)
    dice_scores = []

    for c in range(num_classes):
        pred_c = (preds == c).float()
        gt_c = (ground_truth == c).float()

        if gt_c.sum() == 0:
            continue

        intersection = (pred_c * gt_c).sum()

        dice = (2.0 * intersection + 1e-8) / (pred_c.sum() + gt_c.sum() + 1e-8)
        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return torch.mean(torch.stack(dice_scores)).item()

def evaluate_2d(loader, model):
    """
    Evaluate the model on 2D data and return the average Dice score.
    """
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
    """
    Generate and save plots for training loss, validation loss, and Dice scores over epochs.
    """
    plt.figure()
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend([("Training Loss"), ("Validation Loss")])

    plt.savefig("loss_plot.png")

    plt.figure()
    plt.plot(dice_scores, label="Dice Score")
    plt.savefig("dice_plot.png")

def training_2d():
    """
    Train the 2D U-Net model on the training dataset and validate on the validation dataset.
    """
    # Load Data
    train_dataset = dataset.get_training_2d()
    val_dataset = dataset.get_validation_2d()
    print("data loaded")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Set model, criterion, and optimizer
    model = modules.TwoD_UNet().to(DEVICE)
    img, gt = train_dataset[0]
    print(f"Image shape: {img.shape}, Ground truth shape: {gt.shape}")
    print(torch.unique(gt))
    print("D")
    thresholds = train_dataset.get_proportion().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=thresholds)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_dice = 0.0
    training_losses = []
    validation_losses = []
    dice_scores = []
    print("traing initalised")
    with open("training_log.txt", "a") as log_file:
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()

            train_loss = train_epoch_2d(train_loader, model, optimizer, criterion)
            val_loss = validate_2d(val_loader, model, criterion)
            val_dice = evaluate_2d(val_loader, model)

            epoch_time = time.time() - start_time

            training_losses.append(train_loss)
            validation_losses.append(val_loss)
            dice_scores.append(val_dice)

            log_file.write(
                f"Epoch {epoch+1:03d} "
                f"Train Loss {train_loss:.4f} "
                f"Val Loss {val_loss:.4f} "
                f"Dice {val_dice:.4f} "
                f"Time {epoch_time:.2f}s\n"
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
    return thresholds

def validate_2d(loader, model, criterion):
    """
    Validate the model on 2D data and return the average loss.
    """
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

def test_2d(thresholds):
    """
    Test the 2D U-Net model on the testing dataset and log the results.
    """
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

    running_class_dice = {
        c: 0.0 for c in range(6)
    }
    running_class_iou = {
        c: 0.0 for c in range(6)
    }
    class_count = {
        c: 0 for c in range(6)
    }

    best_dice = -1
    worst_dice = 2
    best_cases = []
    worst_cases = []
    inference_times = []
    with torch.no_grad():

        for image, ground_truth in test_loader:

            image = image.to(DEVICE)
            ground_truth = ground_truth.to(DEVICE)

            start_time = time.time()
            output = model(image)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            class_dice = dice_per_class(output, ground_truth, num_classes=6)
            class_iou = iou_per_class(output, ground_truth, num_classes=6)
            prediction = torch.argmax(output, dim=1)

            with open("training_log.txt", "a") as log_file:
                for c in range(6):
                    fraction = (prediction == c).float().mean().item()

                    log_file.write(
                        f"Class {c} Fraction: {fraction:.4f}\n"
                    )

                for cls, dice_score in class_dice.items():
                    running_class_dice[cls] += dice_score
                    class_count[cls] += 1
                    running_class_iou[cls] += class_iou[cls]

                dice = dice_2d(
                                output,
                                ground_truth
                            )
                
                if dice >= best_dice:
                    best_dice = dice
                    best_cases.append((image.cpu(), prediction.cpu(), ground_truth.cpu()))
                if dice <= worst_dice:
                    worst_dice = dice
                    worst_cases.append((image.cpu(), prediction.cpu(), ground_truth.cpu()))
            
              

            total_dice += dice

            count += 1

    if best_case is not None:
        count = 0
        for best_case in best_cases:
            if count >= 5:
                break
            best_image, best_prediction, best_ground_truth = best_case
            gen_plotting_2d(
                best_image,
                best_prediction,
                best_ground_truth,
                f"best_{count}"
            )
            count += 1

        
    if worst_cases is not None:
        count = 0
        for worst_case in worst_cases:
            if count >= 5:
                break
            worst_image, worst_prediction, worst_ground_truth = worst_case
            gen_plotting_2d(
                worst_image,
                worst_prediction,
                worst_ground_truth,
                "worst"
            )

    # Log results
    with open("training_log.txt", "a") as log_file:
        log_file.write(
            "\nPer-Class Dice Scores\n"
        )
        overall_iou = 0.0
        valid_class_count = 0
        for c in range(6):

            if class_count[c] > 0:

                avg = (
                    running_class_dice[c]
                    / class_count[c]
                )
                overall_iou += (
                    running_class_iou[c]
                    / class_count[c]
                )
                valid_class_count += 1

                log_file.write(
                    f"class {c}: "
                    f"{avg:.4f}\n"
                )

                avg_iou = (
                    running_class_iou[c]
                    / class_count[c]
                )

                log_file.write(
                    f"class {c} IoU: "
                    f"{avg_iou:.4f}\n"
                )
        log_file.write(
            f"Overall IoU: "
            f"{overall_iou / valid_class_count:.4f}\n"
        )
    average_dice = total_dice / count
    with open("training_log.txt", "a") as log_file:
        log_file.write(
            f"Average Dice Score: {average_dice:.4f}\n"
        )

    average_inference_time = sum(inference_times) / len(inference_times)
    with open("training_log.txt", "a") as log_file:
        log_file.write(
            f"Average Inference Time per Image: {average_inference_time:.4f} seconds\n"
        )

    log_file.close()

def gen_plotting_2d(image, combined, combined_truth, count):
    """
    Generate and save plots for the given image, predicted segmentation, and ground truth segmentation.
    """
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
        Patch(color=plt.cm.tab10(0), label="Background"),
        Patch(color=plt.cm.tab10(2), label="Body"),
        Patch(color=plt.cm.tab10(5), label="Bone"),
        Patch(color="grey", label="Bladder"),
        Patch(color="cyan", label="Prostate")
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























BATCH_SIZE = 1
NUM_EPOCHS = 1


root3d_data = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
root3d_truth = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

root3d_data = "C:/Users/hawki/Uni/2026/MRI/3D/semantic_MRs"
root3d_truth = "C:/Users/hawki/Uni/2026/MRI/3D/semantic_labels_only"
# 3D model
def train_epoch_3d(loader, model, optimiser, criterion):
    """
    Train the model for one epoch on 3D data.
    """
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
    """
    Validate the model on 3D data and return the average loss.
    """
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

def dice_3d(preds, ground_truth, num_classes=6):
    """
    Calculate the Dice coefficient for 3D segmentation.
    """
    preds = torch.argmax(preds, dim=1)

    dice_scores = []

    for c in range(num_classes):

        pred_c = (preds == c).float()
        gt_c = (ground_truth == c).float()

        # Skip classes not present in ground truth
        if gt_c.sum() == 0:
            continue

        intersection = (pred_c * gt_c).sum()

        dice = (2.0 * intersection + 1e-8) / (pred_c.sum() + gt_c.sum() + 1e-8)

        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return torch.mean(torch.stack(dice_scores)).item()

def evaluate_3d(loader, model):
    """
    Evaluate the model on 3D data and return the average Dice score.
    """
    model.eval()
    dice_total = 0.0
    count = 0
    with torch.no_grad():
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
    """
    Generate and save plots for training loss, validation loss, and Dice scores over epochs for 3d data
    """
    plt.figure()
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend([("Training Loss"), ("Validation Loss")])

    plt.savefig("loss_plot_3d.png")

    plt.figure()
    plt.plot(dice_scores, label="Dice Score")
    plt.savefig("dice_plot_3d.png")



class model_3d():
    """
    Class to handle the training, validation, and testing of a 3D U-Net model.
    """
    def __init__(self, path1=root3d_data, path2=root3d_truth):
        super().__init__()
        self.unet = modules.ThreeD_UNet()
        tr, va, te = dataset.get_3d_data(path1=path1, path2=path2)
        self.train = tr
        self.val = va
        self.test = te


    def get_data_train(self):
        return self.train
    def get_data_val(self):
        return self.val 
    def get_data_test(self):
        return self.test

    def train_model(self):
        """
        Train the 3D U-Net model on the training dataset and validate on the validation dataset.
        """
        train_loader = torch.utils.data.DataLoader(self.train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val, batch_size=BATCH_SIZE, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
        model = modules.ThreeD_UNet().to(DEVICE)
        thresholds = self.train.get_proportion().to(DEVICE)
        print(f"Thresholds: {thresholds}")
        criterion = nn.CrossEntropyLoss(weight=thresholds)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        
        best_dice = 0.0
    
        training_losses = []
        validation_losses = []
        dice_scores = []
        with open("training_log3d.txt", "w") as log_file:
            for epoch in range(NUM_EPOCHS):
                start_time = time.time()
                train_loss = train_epoch_3d(train_loader, model, optimizer, criterion)
                val_loss = validate_3d(val_loader, model, criterion)
                val_dice = evaluate_3d(val_loader, model)
                epoch_time = time.time() - start_time
        
                training_losses.append(train_loss)
                validation_losses.append(val_loss)
                dice_scores.append(val_dice)
        
                log_file.write(
                    f"Epoch {epoch+1:03d} "
                    f"Train Loss {train_loss:.4f} "
                    f"Val Loss {val_loss:.4f} "
                    f"Epoch Time {epoch_time:.2f}s "
                    f"Dice {val_dice:.4f}\n"
                )
        
                if val_dice > best_dice:
        
                    best_dice = val_dice
        
                    torch.save(
                        model.state_dict(),
                        "best_unet_3d.pth"
                    )
                    
                    log_file.write(
                        f"Saved model (Dice={val_dice:.4f})\n"
                    )
                print("epoch done")
        # print(model.sizes)
        make_plots_3d(training_losses, validation_losses, dice_scores)

    def test_model(self):
        """
        Test the 3D U-Net model on the testing dataset and log the results.
        """
        test_data = self.test
        
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False
        )
    
        model = modules.ThreeD_UNet(
            num_classes=6
        ).to(DEVICE)
    
        model.load_state_dict(
            torch.load(
                "best_unet_3d.pth",
                map_location=DEVICE
            )
        )
    
        model.eval()
    
        total_dice = 0.0
        count = 0
        running_class_dice = {
            c: 0.0 for c in range(6)
        }

        class_count = {
            c: 0 for c in range(6)
        }

        running_class_iou = {
            c: 0.0 for c in range(6)
        }

        inference_time = []
        best_dice = -1
        worst_dice = 2

        best_cases = []
        worst_cases = []

        with torch.no_grad():
            for image, ground_truth in test_loader:
    
                image = image.to(DEVICE)
                ground_truth = ground_truth.to(DEVICE)

                start_time = time.time()
                output = model(image)
                elapsed_time = time.time() - start_time
                inference_time.append(elapsed_time)

                class_dice = dice_per_class(
                    output,
                    ground_truth,
                    num_classes=6
                )

                class_iou = iou_per_class(
                    output,
                    ground_truth,
                    num_classes=6
                )

                for cls, score in class_dice.items():
                    running_class_dice[cls] += score
                    class_count[cls] += 1
                for cls, score in class_iou.items():
                    running_class_iou[cls] += score
 
                dice = dice_3d(
                    output,
                    ground_truth
                )

                mid = image.shape[2] // 2
                    
                mri_slice = image[0, 0, mid].cpu().numpy()

                gt_slice = ground_truth[0, mid].cpu().numpy()
                pred = torch.argmax(output, dim=1)
                pred_slice = pred[0, mid].cpu().numpy()

                if dice > best_dice:
                    best_cases.append(
                        (
                            mri_slice.copy(),
                            gt_slice.copy(),
                            pred_slice.copy()
                        )
                    )

                if dice < worst_dice:
                    worst_cases.append(
                        (
                            mri_slice.copy(),
                            gt_slice.copy(),
                            pred_slice.copy()
                        )
                    )
    
                total_dice += dice
                count += 1
    
                    
        if best_cases:
            count = 0
            for best_case in best_cases:
                if count >= 5:
                    break
                gen_ploting(
                    f"best_{count}",
                    best_case[0],
                    best_case[1],
                    best_case[2]
                )
                count += 1


        if worst_cases:
            count = 0
            for worst_case in worst_cases:
                if count >= 5:
                    break
                gen_ploting(
                    f"worst_{count}",
                    worst_case[0],
                    worst_case[1],
                    worst_case[2]
                )
                count += 1
        
        avg_time = sum(inference_time) / len(inference_time)
        with open("training_log3d.txt", "a") as log_file:
            log_file.write("\nPer-Class Dice Scores\n")
            overall = 0
            valid = 0
            for c in range(6):
                if class_count[c] > 0:
                    avg = (
                        running_class_dice[c]
                        / class_count[c]
                    )
                    overall += avg
                    valid += 1
                    log_file.write(
                        f"class {c}: "
                        f"{avg:.4f}\n"
                    )

            log_file.write(
                f"\nOverall Dice Score: {overall / valid if valid > 0 else 0.0:.4f}\n"
            )


            log_file.write("\nPer-Class IoU Scores\n")
            overall = 0
            valid = 0
            for c in range(6):
                if class_count[c] > 0:
                    avg = (
                        running_class_iou[c]
                        / class_count[c]
                    )
                    overall += avg
                    valid += 1
                    log_file.write(
                        f"class {c}: "
                        f"{avg:.4f}\n"
                    )
            overall = overall / valid if valid > 0 else 0.0
            log_file.write(
                f"Overall: {overall:.4f}\n"
            )

            log_file.write(
                f"\nAverage Inference Time: "
                f"{avg_time:.4f}s\n"
            )
    

def gen_ploting(name, mri_slice, gt_slice, pred_slice):
    """
    Generate and save plots for the given MRI slice, ground truth segmentation, and predicted segmentation.
    """
    plt.figure(figsize=(16,5))
        
    plt.subplot(1,5,1)
    plt.imshow(
        mri_slice,
        cmap="gray"
    )
    plt.title("MRI")
    plt.axis("off")

    plt.subplot(1,5,2)
    plt.imshow(
        gt_slice,
        cmap="tab10",
        vmin=0,
        vmax=5
    )
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,5,3)
    plt.imshow(
        pred_slice,
        cmap="tab10",
        vmin=0,
        vmax=5
    )
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1,5,4)
    plt.imshow(
        mri_slice,
        cmap="gray"
    )

    plt.imshow(
        pred_slice,
        cmap="tab10",
        alpha=0.4,
        vmin=0,
        vmax=5
    )
    plt.title("Overlay")
    plt.axis("off")

    diference = gt_slice - pred_slice
    plt.subplot(1,5,5)
    plt.imshow(
        diference,
        cmap="bwr",
        vmin=-5,
        vmax=5
    )
    plt.title("Difference (GT - Pred)")
    plt.axis("off")

   

    legend_items = [
        Patch(color=plt.cm.tab10(0), label="Background"),
        Patch(color=plt.cm.tab10(2), label="Body"),
        Patch(color=plt.cm.tab10(5), label="Bone"),
        Patch(color="grey", label="Bladder"),
        Patch(color="cyan", label="Prostate")
    ]

    plt.legend(
        handles=legend_items,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        f"prediction3d_{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()