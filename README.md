# Enhancing MRI Image Analysis and Report Generation by Fine-Tuning Pre-trained Vision Language Models

**Authors:** Bowen Xia, Li Jin — The Harker School, San José, CA; Department of Kinesiology, San José State University, San José, CA

---

## Motivation

Knee joint injuries, particularly ACL tears, are among the most common injuries in athletes. Magnetic Resonance Imaging (MRI) is the standard approach for visualizing and diagnosing these injuries. However, the intricate and voluminous nature of MRI data poses significant challenges for radiologists who must carefully analyze each image. By leveraging generative AI, this project aims to automate and improve the accuracy of MRI image analysis, providing radiologists with powerful tools to aid in diagnosis.

## Goal

Integrate AI into the diagnostic process to streamline and improve ACL injury detection for radiologists.

---

## Workflow

### 1. Data Collection and Preprocessing

The dataset consists of 731 unique MRI scans compiled on Kaggle, each with a JSON file containing the coordinates of consecutive 2D slices — 542 normal and 189 with tears (at ratios of 74.1%, 19.8%, and 21.1%). A metadata file defines the region of interest for the coordinates, identifying the 22 slices that contain the most relevant and clear view of the ACL. It also includes X and Y ROI values used for bounding box segmentation. For each study, the 22 slices are extracted according to ROI depth and compiled into a labeled table.

### 2. MRI Segmentation with MedSAM

Rather than using traditional CNNs (which were previously applied to classify ACL tears), this project uses an improved pipeline: **MedSAM** (Medical Segment Anything Model) segments each MRI image and highlights the potential areas of interest for doctors. For a torn ACL, MedSAM highlights the entire region due to the absence of the defined normal ACL structure.

### 3. Report Generation with GPT-4 Vision (RAG)

The segmented MRI image is passed to **ChatGPT with a vision prompt** as a condition to generate a radiological report. The system uses **RAG (Retrieval-Augmented Generation)** with medical terminology to improve the accuracy of the generated text. A sample prompt instructs the model to give a brief two-to-three sentence report on the findings and features visible in the image.

For a **torn ACL**, the model produces output such as: *"The MRI image demonstrates a complete tear of the anterior cruciate ligament (ACL). The torn ligament is visible as irregular, disrupted tissue, with intercondylar notch, with significant signal alteration consistent with injury. Surrounding soft tissue edema and fluid accumulation further confirm the severity of the ACL tear."*

For a **normal ACL**, the output describes a continuous, well-defined ligament with no abnormal signal intensities, edema, or fluid collection.

### 4. Fine-Tuning with IDEFICS 2 (LoRA)

To further improve performance, the pre-trained vision-language model is fine-tuned using **LoRA (Low-Rank Adaptation)**. LoRA enables efficient adaptation of the base model by learning low-rank matrix updates, which significantly reduces computational cost and memory requirements during fine-tuning while maintaining the model's performance.

### 5. Results

The fine-tuned model produces accurate, clinically relevant predictions. For example, a torn ACL prediction reads: *"The anterior cruciate ligament (ACL) shows complete disruption, indicative of a full-thickness tear. The ligament fibers are discontinuous, with loss of normal tension and orientation. There is significantly increased signal intensity at the site of the tear, confirming a complete rupture."*

Evaluated metrics include accuracy, precision, recall, and AUC — with EfficientNet achieving strong classification performance across normal, partial, and complete tear categories.

---

## Tech Stack

- **MedSAM** — MRI region segmentation
- **GPT-4 Vision + RAG** — radiological report generation
- **IDEFICS 2 + LoRA (IDEFICS 2)** — fine-tuned vision-language model for prediction
- **EfficientNet** — baseline CNN classification comparison
- **Kaggle MRI Dataset** — 731 knee MRI scans with ACL labels
