Task 2.2 Segmentation: Hip MRI Anatomical Structure Segmentation

Comparison of 2D and 3D U-Net Architectures for Hip MRI Segmentation




Problem
The beneficiaries of the proposed system are patients undergoing hip MRI imaging and the medical professionals responsible for interpreting the resulting scans. The intended end users are clinicians who can utilise the segmentation tool to assist in identifying anatomical structures, verifying suspected regions of concern, and detecting segmentation boundary leakage that may indicate areas requiring further investigation.

The objective of this project is to perform semantic segmentation of pelvic MRI scans using U-Net based deep learning architectures. The task involves automatically assigning each pixel or voxel within an MRI scan to one of several anatomical structures, allowing clinically relevant regions to be isolated from surrounding tissue. The target structures include the body, bone, bladder, colon, and prostate, together with a background class.

U-Net is a convolutional neural network architecture specifically designed for image segmentation. As shown in Figure 1, the network consists of a contracting encoder path and an expanding decoder path connected through skip connections. The encoder extracts increasingly abstract features from the input image, while the decoder reconstructs the segmentation map from these learned features. The skip connections preserve spatial information lost during down-sampling, enabling precise localisation of boundaries and improving accuracy.

A two-dimensional U-Net architecture was implemented as a baseline model and comparisons will be made with a 3D U-Net structure using the Sorensen-Dice score as an accuracy metric and compute times and memory consuption will also be used for comparing the effectiveness/tradeoofs of the models. 

Figure 1: Unet structures




Feasability report
Note: This report represents the ambitions of the project but was not fully implemented



To investigate whether volumetric context improves segmentation performance, a three-dimensional U-Net architecture was also implemented. The 3D model follows the same encoder-decoder structure as the baseline model but replaces two-dimensional operations with their three-dimensional equivalents, allowing neighbouring MRI slices to contribute contextual information during segmentation. Unlike the 2D model, which operates on individual slices, the 3D U-Net processes complete MRI volumes.




Dependencies and Reproducability
The implementation was developed in Python using the PyTorch deep learning framework. Medical image volumes and segmentation masks stored in NIfTI format were loaded using NiBabel, while NumPy was used for numerical processing and preprocessing operations. Visualisations of predictions, overlays, training curves, and failure cases were produced using Matplotlib. Training was performed using the Adam optimiser with GPU acceleration provided through CUDA-enabled PyTorch operations when available.

All source code was maintained within a Git repository to ensure experiments could be reproduced and tracked throughout development. Initial proof-of-concept experiments were conducted locally using CPU execution and reduced datasets, while final experiments were executed on the Rangpur high-performance computing cluster using NVIDIA A100 GPUs.

Training was performed with a batch size of 2 for 50 epochs using the Adam optimiser with a learning rate of 0.001. A combined Binary Cross-Entropy and Dice loss function was selected, as Dice similarity coefficient (DSC) was the primary evaluation metric used throughout the project.




Example (in, outm visual plots)
The input to the proposed system consists of pelvic MRI scans represented either as individual two-dimensional slices or as complete three-dimensional MRI volumes. Associated ground-truth segmentation masks are provided for six anatomical classes: Background, Body, Bone, Bladder, Colon, and Prostate.

Representative examples are automatically generated during evaluation, together with the highest-performing and lowest-performing predictions. These visualisations are used to qualitatively assess segmentation accuracy and identify common failure cases such as boundary leakage and incorrect anatomical classification.




Pre processing
MRI images were loaded from the provided NIfTI (.nii) files using the supplied dataset conversion utilities. Image intensities were normalised prior to training to reduce variation between scans and improve training stability. For the 2D implementation, segmentation labels were converted into a fixed six-channel categorical representation to support the Binary Cross-Entropy and Dice loss training objective. In contrast, the 3D segmentation masks were retained as integer class labels, as required by the Cross Entropy loss function.

To ensure consistent network inputs, images were resized to a common resolution. The final input dimensions for the 2D model were 256 × 144 pixels, while the 3D model operated on volumes of 144 × 256 × 256 voxels. Ground-truth segmentations were resized using nearest-neighbour interpolation to preserve class labels, while MRI images used interpolation methods suitable for continuous image data.

The provided 2D dataset was supplied with predefined training, validation, and testing partitions. The 3D dataset required manual partitioning and was split at the patient level using a 70%, 15%, 15% training, validation, and testing allocation with random seed of 42. This split was selected to maximise the amount of training data available while maintaining sufficient validation and testing samples for model selection and final evaluation. A patient-level split also prevents information leakage between subsets by ensuring that different scans from the same patient cannot appear in multiple partitions.




Open Research Dilemma:
Benchmarking

Resource profiling

Error autopsies

Recomendations and trade offs

AI disclosure
GitHub Copilot for auto complete
Copilot for Unet model structure - checked by comparison with other models found during research Ref ()
Copilot was used for editing the report - creating sentences given dot points or specification I wrote

References:

