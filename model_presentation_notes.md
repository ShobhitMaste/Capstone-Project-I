# FedTrap: EfficientNet-B0 Model Architecture & Pipeline
*Presenter Notes for Review 3*

## 1. Dataset Engineering & Merging
To create a robust real-world model, we merged two distinct datasets to increase our class diversity and overall image volume.
* **Source 1:** Agricultural Pests Image Dataset (12 classes, sourced from Flickr via API)
* **Source 2:** Insects Recognition Dataset (5 classes, scraped from Google and iStock)
* **Merging Strategy:** We algorithmically mapped the classes to prevent duplication and taxonomy errors. For example, "Ladybird" (Insects Recognition) was merged into the "Beetle" class (Agricultural Pests) because ladybirds are part of the Coleoptera order. 
* **Final Result:** 15 distinct classes with ~9,900 verified images.

## 2. Image Preprocessing & Augmentation
To make the model resilient to the unpredictable lighting and angles of a live camera feed inside the trap, we heavily augmented the training data:
* **Training Augmentations:**
  * `RandomResizedCrop(224)` (simulates bugs at different distances)
  * `RandomHorizontalFlip` (rotation invariance)
  * `ColorJitter` (brightness, contrast, and saturation variations to simulate different times of day)
  * `RandomRotation(15 degrees)`
* **Normalization:** All images are normalized using the standard ImageNet statistics `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.

## 3. Model Architecture: EfficientNet-B0
We pivoted to **EfficientNet-B0** because it offers the perfect balance between high accuracy and computational efficiency for Edge AI (Raspberry Pi).
* **Parameters:** ~5.3 million parameters (highly optimized for mobile/edge).
* **Transfer Learning:** We initialized the model with ImageNet-1K pre-trained weights to leverage its existing feature-extraction capabilities.
* **Custom Head:** We removed the original 1000-class head and replaced it with a `Dropout(0.2)` layer followed by a `Linear` layer mapping to our 15 insect classes.

## 4. Two-Phase Training Strategy
We utilized a two-phase transfer learning approach to prevent "catastrophic forgetting" of the pre-trained weights:
1. **Phase 1 (10 Epochs):** We froze the entire EfficientNet backbone. We only trained our new 15-class classifier head. This allows the random weights in the new head to stabilize.
2. **Phase 2 (20 Epochs):** We unfroze the top 3 blocks of the EfficientNet backbone and lowered the learning rate by 10x. This allows the model to fine-tune its feature extraction specifically for insect shapes and textures.

## 5. Overcoming Class Imbalance
Because we merged datasets, some classes had 1,000+ images while others had 300. To prevent the model from becoming biased toward the majority classes, we implemented **Weighted Cross-Entropy Loss**. The algorithm mathematically penalized the model more heavily for getting rare insects wrong, forcing it to learn all 15 classes equally.

## 6. Federated Learning Simulation (FedAvg)
To prove our FedTrap concept for Review 3, we successfully simulated Federated Learning:
* We partitioned the dataset into 2 simulated clients using a **Non-IID Dirichlet distribution**. This proves our model can learn even when one trap sees entirely different insects than another trap.
* We utilized the **FedAvg (Federated Averaging)** algorithm, running 3 local epochs per client before aggregating the weights globally. 
* We successfully extracted the `federated_best.pt` global model, proving our PyTorch architecture is 100% ready for physical multi-Pi deployment using the Flower framework in Review 4.
