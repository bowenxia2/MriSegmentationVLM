# Enhancing MRI Image Analysis and Report Generation by Fine-Tuning Pre-trained Vision Language Models

**Authors:** Bowen Xia, Li Jin | The Harker School, San José, CA; Department of Kinesiology, San José State University, San José, CA

---

## Motivation

Knee joint injuries, particularly ACL tears, are among the most common injuries in athletes. Magnetic Resonance Imaging (MRI) is the standard approach for visualizing and diagnosing these injuries. However, the intricate and voluminous nature of MRI data poses significant challenges for radiologists who must carefully analyze each image. By leveraging generative AI, this project aims to automate and improve the accuracy of MRI image analysis, giving radiologists more powerful tools to support their diagnosis.

## Goal

Integrate AI into the diagnostic process to streamline and improve ACL injury detection for radiologists.

---

## Workflow

### 1. Data Collection and Preprocessing

The dataset consists of 731 unique MRI scans compiled on Kaggle, each with a JSON file containing the coordinates of consecutive 2D slices (542 normal and 189 with tears, at ratios of 74.1%, 19.8%, and 21.1%). A metadata file defines the region of interest, identifying the 22 slices that contain the most relevant and clear view of the ACL. It also includes X and Y ROI values used for bounding box segmentation. For each study, the 22 slices are extracted according to ROI depth and compiled into a labeled table.

### 2. MRI Segmentation with MedSAM

Rather than using traditional CNNs, this project uses **MedSAM** (Medical Segment Anything Model) to segment each MRI image and highlight potential areas of interest for doctors. MedSAM is prompted with the ROI bounding box coordinates from the metadata, which tells the model where to focus. It then produces a segmentation mask over the ACL region and draws a blue bounding box on the output image. For a torn ACL, MedSAM highlights the entire region due to the absence of the defined normal ACL structure.

After segmentation, the blue bounding box drawn by MedSAM is extracted to crop the image to the region of interest. No additional model inference happens in this step. The intelligence comes entirely from MedSAM; the cropping is purely mechanical color detection.

### 3. Report Generation with GPT-4 Vision (RAG)

The segmented MRI image is passed to **GPT-4o with a vision prompt** to generate a radiological report. The system uses **RAG (Retrieval-Augmented Generation)** with medical terminology to improve the accuracy of the generated text. The prompt instructs the model to give a brief two-to-three sentence report on the findings and features visible in the image.

For a **torn ACL**, the model produces output such as: *"The MRI image demonstrates a complete tear of the anterior cruciate ligament (ACL). The torn ligament is visible as irregular, disrupted tissue, with intercondylar notch, with significant signal alteration consistent with injury. Surrounding soft tissue edema and fluid accumulation further confirm the severity of the ACL tear."*

For a **normal ACL**, the output describes a continuous, well-defined ligament with no abnormal signal intensities, edema, or fluid collection.

### 4. EfficientNet CNN Classification

As part of the workflow, an **EfficientNet** CNN model was used to classify MRI slices into normal, partial tear, and complete tear categories. This provided a strong baseline for evaluating classification performance, with metrics including accuracy, precision, recall, and AUC. The code for this step is not included in this repository.

### 5. Fine-Tuning with LLaVA (LoRA)

To further improve performance, **LLaVA-1.5-13b** is fine-tuned using **LoRA (Low-Rank Adaptation)** and DeepSpeed for multi-GPU training. LoRA allows for efficient adaptation of the pre-trained model by learning low-rank matrix updates, which significantly reduces computational cost and memory requirements during fine-tuning while maintaining model performance.

### 6. Results

The fine-tuned model produces accurate, clinically relevant predictions. For example, a torn ACL prediction reads: *"The anterior cruciate ligament (ACL) shows complete disruption, indicative of a full-thickness tear. The ligament fibers are discontinuous, with loss of normal tension and orientation. There is significantly increased signal intensity at the site of the tear, confirming a complete rupture."*

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `mri_dataset.ipynb` | Initial data processing notebook used to work with the raw Kaggle volumetric MRI scans. Loads the metadata CSV, resolves file paths across volume subfolders, and extracts individual 2D slices from the volumetric pickle files for use in the rest of the pipeline. |
| `medsam_all_images.ipynb` | Loads volumetric MRI pickle files, extracts the relevant 2D slices per exam using ROI metadata, runs MedSAM segmentation with bounding box prompts, and saves segmented images to disk. |
| `mri_boundingbox_extraction.ipynb` | Takes a MedSAM-segmented image (which has a blue bounding box drawn on it) and uses OpenCV color thresholding to detect and crop to that region. No ML involved; this is purely classical CV used as a post-processing utility. |
| `data_prep_full_hf_mri.ipynb` | Sends each segmented MRI image to GPT-4o with a diagnosis-conditioned prompt to generate radiological report captions. Formats the resulting image and caption pairs into a structured dataset with a stratified train/test split for fine-tuning. |
| `Final_mri_llava_finetune_new.ipynb` | End-to-end LLaVA-1.5-13b fine-tuning pipeline. Installs LLaVA and DeepSpeed, runs LoRA fine-tuning across multiple GPUs with Weights and Biases tracking, merges the LoRA adapter weights into the base model, and deploys the result via a Gradio interface. |

---

## Tech Stack

- **MedSAM** for MRI region segmentation
- **GPT-4o + RAG** for radiological report generation
- **LLaVA-1.5-13b + LoRA** for fine-tuned vision-language model prediction
- **EfficientNet** for baseline CNN classification (code not included)
- **Kaggle MRI Dataset** with 731 knee MRI scans and ACL labels
