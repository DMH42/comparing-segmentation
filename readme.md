# Who Would Win: A State-of-the-Art Foundation Model or a Neural Net?

### Comparing the performance of Traditional Deep Neural Networks to Grounded Dino + SAM2

We often hear about the magnificent performance of Foundation Models and how they are going to revolutionize the world of AI. However, as a Machine Learning Engineer with years of experience in the field, I'd say that one of the first lessons we learn is that sometimes a simple approach can perform just as well, or even better, than a shiny new model. Let's find out to what extent that is true!

#### Objective:

In this article we will discuss the usage of Foundation Models for the task of image segmentation and compare them to a more traditional approach of fine tunning using deep learning.

##### Code Walkthrough

[Here](https://github.com/DMH42/comparing-segmentation/blob/main/ComparingSegmentation.ipynb) you can find the Notebook that we will be walking through in this post.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DMH42/comparing-segmentation/blob/main/ComparingSegmentation.ipynb)

For this article we will be using [this](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset) Kaggle Dataset

#### Introduction:

**What is Image Segmentation?**
Image segmentation is a task where we identify an object in an image and then trace the boundary around the object. If you use stickers on an [iPhone](https://support.apple.com/guide/iphone/make-stickers-from-your-photos-iph9b4106303/ios) or WhatsApp, then you are already familiar with the task of Image Segmentation!

**Traditional Deep Learning:**
There's a great series on [Deep Learning by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) if you are not too familiar with Deep Learning or if you'd like to have a refresher. Deep learning is the foundation of the current AI revolution, and it can be helpful to return to the basics when tackling a new problem.
Traditional Deep learning is a good starting point that can be used to benchmark your solutions. Entire books and posts can be written on how to come up with a good deep learning architecture. Fortunately, for image segmentation, we are already aware of effective architectures.
Since we have a relatively large dataset of roughly 1k images, we can confidently take the approach of training a neural network. There are two deep learning approaches that we will take: train a [UNet](https://www.youtube.com/watch?v=NhdzGfB1q74) and fine-tune a ResNet. We will dive deeper into both of them later in the post.

**The State of The Art: Foundation Models**

The rise of the [transformer](https://arxiv.org/pdf/1706.03762) architecture has ushered in a drastic change in the machine learning space. To put it simply, foundation models are large deep learning models that have been trained on copious amounts of data. The requirements for training these models are so large that it is prohibitively expensive for everyone but a small handful of companies. The line between a regular pre-trained model and a foundation model is a bit blurry, especially as the models get released by companies and the hardware becomes more capable however, one key distinction would be the capability of foundation models to perform Zero Shot inference which is an emergent capability of these models as compared to smaller pretrained models.

**What do these approaches have in common?**
The most important commonality of these approaches is leveraging pre-trained models, as that allows for faster results and a small or nonexistent training phase. This is particularly beneficial when we are limited in either data or resources.

**When to choose Foundation Models vs Traditional Deep Learning?**

In general, if you have a lot of data and the task is well defined then it is often the case that a traditional deep learning approach will yield better results. However, if you do not have enough data or the task is not well defined then it is often the case that a foundation model will be able to give you better results.

Another factor to consider is the time and resources you have available. If you need to get results quickly and do not have the resources to train a model from scratch, then a foundation model is often the best choice. However, if you have the time and resources to train a model from scratch, then a traditional deep learning approach will often yield better results.

Finally if you are developing a production system, you have to think of the inference time and the resources that you have available. Foundation models are often very large and require a lot of resources to run, which can make them impractical for production systems. In this case, a traditional deep learning approach is often the best choice.

### Implementation

**Setting up the project:**

```
!pip install kaggle
!mkdir -p /root/.config/kaggle
!sudo cp kaggle.json ~/.config/kaggle/
!sudo chmod 600 ~/.config/kaggle/kaggle.json
!pip install torch torchvision segmentation-models-pytorch albumentations opencv-python tqdm
!kaggle datasets download -d tapakah68/segmentation-full-body-mads-dataset
!unzip segmentation-full-body-mads-dataset.zip
```

## Deep Learning

UNet
For this scenario, we will be training both a UNet and a pretrained ResNet. The UNet is a very commonly used architecture for the Image Segmentation task, particularly starting with the application for [Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). In this case, we will be implementing the UNet and also training it from scratch using the data from the dataset. The purpose of this is to evaluate what the "worst-case scenario" for performance would be. We'll be implementing the UNet through PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        bn = self.bottleneck(self.pool4(d4))

        up4 = self.up4(bn)
        up4 = torch.cat([up4, d4], dim=1)
        dec4 = self.dec4(up4)

        up3 = self.up3(dec4)
        up3 = torch.cat([up3, d3], dim=1)
        dec3 = self.dec3(up3)

        up2 = self.up2(dec3)
        up2 = torch.cat([up2, d2], dim=1)
        dec2 = self.dec2(up2)

        up1 = self.up1(dec2)
        up1 = torch.cat([up1, d1], dim=1)
        dec1 = self.dec1(up1)

       return self.final_conv(dec1)
```

**Importing Packages:**

```python
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
```

**Setting up the directories**

```python
image_dir = './segmentation_full_body_mads_dataset_1192_img/segmentation_full_body_mads_dataset_1192_img/images'
mask_dir = './segmentation_full_body_mads_dataset_1192_img/segmentation_full_body_mads_dataset_1192_img/masks'
```

**Creating the Segmentation Dataset**

```python
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # Convert to RGB
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask
```

**Defining the data augmentations through the use of the Albumentations library:**

```python
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])
```

Setting up the datasets:

```python
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42
)

train_dataset = SegmentationDataset(train_imgs, train_masks, transforms=train_transform)
val_dataset = SegmentationDataset(val_imgs, val_masks, transforms=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)
```

**Adding the model to the device**

```python
model = UNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

**Setting up the Binary Cross-Entropy Loss Function and the Optimizer:**

```python
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

Defining the evaluation function
The evaluation method that we are using is the [Intersection Over Union](https://huggingface.co/learn/computer-vision-course/en/unit6/basic-cv-tasks/segmentation#how-to-evaluate-a-segmentation-model). It is common to use this metric (the ratio between the intersection of the predicted and true mask) divided by the union of the two. The closer the result is to 1 then then the closer the two masks are to being the same.

What we are doing here is to convert the segmentation masks into boolean arrays to then be able to do logical operations on them (in this case, intersection and union). Then we compare the ratio between the intersection and the union.

```python
def iou_score(preds, targets, threshold=0.5):
    predicted_mask = preds > threshold
    predicted_mask = predicted_mask.bool()
    true_mask = targets.bool()
    intersection = (predicted_mask & true_mask).sum(dim=(1, 2, 3))
    union = (predicted_mask | true_mask).sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()
```

**We'll define the functions needed to train the model for one epoch and evaluate the model:**

```python
scaler = GradScaler()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            preds = model(imgs)
            loss = criterion(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_iou = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            preds = model(imgs)
            preds = torch.sigmoid(preds)
            total_iou += iou_score(preds, masks)
    return total_iou / len(loader)
```

**Finally, we train the model and evaluate the model:**

```python
epochs = 25
train_losses = []
val_ious = []
for epoch in range(epochs):
    print("starting to train")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_iou = evaluate(model, val_loader)
    train_losses.append(train_loss)
    val_ious.append(val_iou)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")
```

The performance we got from training the UNet from scratch was **0.9167**, which is a decent level of performance, from not having the advantages of an excessively large dataset or even pre-training.

### ResNET

We use the ResNET in this case to evaluate how good the performance of the model would be if we were to have the advantages of a model with existing weights that we can use as a pretrained checkpoint to start from.

**Importing the segmentation model [library](https://pypi.org/project/segmentation-models-pytorch/):**

```python
import segmentation_models_pytorch as smp
```

**Using the library to import the pretrained ResNet:**

```python
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)
```

**Setting up the model for the run by adding it to the GPU and setting up the loss and optimizer:**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

**We finally train the model:**

```python
epochs = 20
train_losses = []
val_ious = []
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_iou = evaluate(model, val_loader)
    train_losses.append(train_loss)
    val_ious.append(val_iou)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")
```

We were able to obtain a final score of **0.9208**, which is a very good performance for a model that was trained on a relatively small dataset. This is a good example of how traditional deep learning approaches can still be very effective for image segmentation tasks, especially when we have enough data to train on.

### SAM2 + GroundingDino

One of the trends that we have been observing is the implementation of the transformer architecture into a plethora of domains. An example of one of this increased prevalence of the transformer architecture is the [Segment Anything Model](https://arxiv.org/pdf/2304.02643) by Meta.

The newest version [SAM2](https://arxiv.org/pdf/2408.00714) has increased performance and capabilities, this is due to the fact that Meta and other similar companies are able to train these models with vast quantities of data, more than what an individual would be able to do on their own computer.

However, one of the shortcomings of this model is that it does not have the capability to receive a text prompt to then segment those objects from the image or video. The reason for this is that the SAM models require a recommendation in the form of a point or a bounding box to be able to then make their segmentation inference.

If we are able to have a model that is able to take in arbitrary text and output a point or a bounding box of those objects in the image then we would be able to have a zero shot image segmentation model! Fortunately, this is exactly what the [GroundingDINO](https://arxiv.org/pdf/2303.05499) model does!

Those two ideas together researchers at IDEA-Research created [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2). While they came up with the idea, I found the implementation of GroundingDino + SAM2 by [Luca Medeiros](https://github.com/luca-medeiros/lang-segment-anything) to be more developer friendly which is the one that I will be using for this article.

**Installing the lang-sam library**

`!pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git`

**Importing the libraries**

```python
from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os
import matplotlib.pyplot as plt
import kaggle
import cv2
model = LangSAM()
text_prompt = "person."
```

**The intersection over union metric for this model:**

```python
def intersection_over_union_metric(predicted_mask, true_mask):
 predicted_mask = predicted_mask.astype(bool)
 true_mask = true_mask.astype(bool)
 intersection = np.logical_and(predicted_mask, true_mask).sum()
 union = np.logical_or(predicted_mask, true_mask).sum()
 if union == 0:
 return 0.0
 iou = intersection / union
 return iou
```

**Picking Function**
We define this small function to find the index of the best mask from the model output.
This function will be relevant later on in the tutorial.

```python
def pick_best(masks, masks_scores, scores)
 max_index = 0
 for i, predicted_mask in enumerate(masks):
 # find the mask that has the highest score
 if masks_scores[max_index] < masks_scores[i]:
 max_index = i
 return max_index
```

**Processing Function**

The following function is used to load the image, obtain the inference results from the model, and use the _pick_best_ function to pick the best mask result and return it.

```python
def evaluate_image(image_file, image_path, mask_path):
   #Load the image
   image_pil = Image.open(image_path).convert("RGB")
   image_np = np.array(image_pil)
   # Load the ground truth mask
   true_mask_pil = Image.open(mask_path).convert("L")
   # Convert to boolean mask
   true_mask_np = np.array(true_mask_pil) > 0
   # Predict the mask using LangSAM
   result = model.predict([image_pil],  [text_prompt])
   scores = result[0]['scores']
   boxes = result[0]['boxes']
   masks = result[0]['masks']
   masks_scores = result[0]['mask_scores']
   # the model could output no masks
   if  len(masks) == 0:
    print(f"No Prediction found for {image_file}")
    return {'image': image_file,  'iou': np.nan}
   if masks is  not  None  and  not(isinstance(masks,  list)):
    if masks_scores.ndim == 0:
     masks_scores = [masks_scores.item()]
    if scores.ndim == 0:
     scores = [scores.item()]
    max_index = pick_best(masks, masks_scores, scores)
    predicted_mask_np = masks[max_index]
    result_iou = evaluation_function(predicted_mask_np, true_mask_np)
    return {'image': image_file,  'iou': result_iou}
```

Defining the evaluation loop
This function is the evaluation loop that we'll use to evaluate the results of the model. We load the files, use the evaluation function we defined previously, and return the aggregated results.

```python
def  evaluate_model_on_test_set(test_images_dir='./test_images',
test_masks_dir='./test_masks',
text_prompt="person.",
evaluation_function=intersection_over_union_metric, debug=True
):
 iou_results = []
 # Ensure the directories exist
 if  not os.path.exists(test_images_dir):
  print(f"Test images directory not found: {test_images_dir}")
 elif  not os.path.exists(test_masks_dir):
  print(f"Test masks directory not found: {test_masks_dir}")
 else:
  image_files = [f for f in os.listdir(test_images_dir)  if f.endswith(('.png',  '.jpg',  '.jpeg'))]
 for image_file in image_files:
  image_path = os.path.join(test_images_dir, image_file)
  mask_path = os.path.join(test_masks_dir, image_file)
  if  not os.path.exists(mask_path):
   print(f"Ground truth mask not found for {image_file}")
   continue
  try:
   iou_results.append(evaluate_image(image_file, image_path, mask_path))
  except Exception as e:
   # if there was an error processing then we insert a NaN value
   if debug: print(f"Error processing {image_file}: {e}")
   iou_results.append({'image': image_file,  'iou': np.nan})
 # Print average IoU
 if debug and len(iou_results) > 0:
  average_iou = np.nanmean([res['iou']  for res in iou_results])
  print(f"\nAverage IoU across test set: {average_iou:.4f}")
 elif debug:
  print("No images were processed.")
 return iou_results
```

**Run the evaluation loop**
We then run the evaluation loop on the dataset.

```python
def make_histogram(iou_results)
 iou_scores = [res['iou'] for res in iou_results if not np.isnan(res['iou'])]
 if len(iou_scores) >= 0:
  plt.figure(figsize=(10, 6))
  plt.hist(iou_scores, bins=25, edgecolor='black')
  plt.title('Distribution of IoU Scores on Test Set')
  plt.xlabel('IoU Score')
  plt.ylabel('Frequency')
  plt.grid(axis='y', alpha=0.75)
  plt.show()
 else:
  print("No valid IoU scores available to plot the histogram.")
```

**Results**

We get an average score of 0.8001 when we run the evaluation loop, which is a decent performance for a task that the model was not fine-tuned for this task as under the hood, two models are run sequentially to obtain the predicted mask.
**Making a histogram**
Oftentimes, when we evaluate models, it is useful to see the distribution of our results. A helpful tool for the task is the use of a histogram.

```python
def make_histogram(iou_results)
 iou_scores = [res['iou'] for res in iou_results if not np.isnan(res['iou'])]
 if len(iou_scores) >= 0:
  plt.figure(figsize=(10, 6))
  plt.hist(iou_scores, bins=25, edgecolor='black')
  plt.title('Distribution of IoU Scores on Test Set')
  plt.xlabel('IoU Score')
  plt.ylabel('Frequency')
  plt.grid(axis='y', alpha=0.75)
  plt.show()
 else:
  print("No valid IoU scores available to plot the histogram.")
```

![](./images/histogram1.png)

As we can see, we have a lot of instances with 0 as their score. The distribution is not continuous and they cluster at 0, this is often indicative of some sort of recurring problem which means that we should dig deeper.

**Exploring the lowest performing images**
We can use this code snippet in order to obtain the worst performing images and see if there is something we can do to improve the performance.

```python
# Sort the iou_results by IoU in ascending order
sorted_iou_results = sorted(iou_results, key=lambda x: x['iou'])

# Get the lowest values and their corresponding image names
lowest_iou_results = sorted_iou_results[:20] # Get the bottom 10

print("Lowest IoU values and corresponding image names:")
for result in lowest_iou_results:
  print(f"Image: {result['image']}, IoU: {result['iou']:.4f}")
```

Unfortunately as it is often the case when working with ML models, we got some false positives. As such, we should figure out what we can do in order to improve the performance.

**Show the masks**
To dig deeper we can go ahead and explore the failed results. We define these functions to use them as helpers in order to show the base image, and the corresponding predicted masks.

```python
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=1)
    ax.imshow(mask_image)

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.5f}", fontsize=18)
        plt.axis('off')
        plt.show()
```

**Lets pick one of the lower performing images**

```python
image_file = "Jazz_Jazz2_C1_00630.png"
image_path = os.path.join(test_images_dir, image_file)
mask_file = image_file
mask_path = os.path.join(test_masks_dir, mask_file)
image_pil = Image.open(image_path).convert("RGB")
image_np = np.array(image_pil)
true_mask_pil = Image.open(mask_path).convert("L")
true_mask_np = np.array(true_mask_pil) > 0
result = model.predict([image_pil], [text_prompt])
scores = result[0]['scores']
boxes = result[0]['boxes']
masks = result[0]['masks']
mask_scores = result[0]['mask_scores']
show_masks(image_pil, masks, scores)
```

In this case, the model correctly identified an object that is shaped like a person but is not a real person.

![](./images/false_positive.png)

**Updating The Picking Function**
We can use both the information from the SAM2 model and the GroundedDino model in order to be able to get a more accurate selection. This modified function will make it so that we only pick the object that both of the scores of the models agree. So instead of just picking the best result from the SAM2 Model (the mask_scores) we use also the best score from the Grounding DINO model.

```python
def pick_best(masks, masks_scores, scores)
	max_index = 0
	for i, predicted_mask in  enumerate(masks):
		# find the mask that has the highest score
		if masks_scores[max_index] < masks_scores[i]:
			if scores[max_index] < scores[i]:
				max_index = i
	return max_index
```

This function now allows us to get this accurate mask:

![](./images/correct_predicted.png)

**Results**
![](./images/histogram2.png)

By implementing this small change on the picking function, we can boost the performance to an average of **0.8830**, which is a much better performance.

As you've seen, it was very straightforward to set up the inference for the model and to start generating results. We needed to do a little bit of debugging in order to boost the performance, but it didn't require a lot more effort. This is an example of how foundation models can help increase the productivity of engineering teams by allowing us to ship quickly.

## Conclusion

Through this walkthrough, you have now seen how to do Zero Shot Image Segmentation through the usage of the SAM2 model and the GroundingDino model. Not only that, but we also explored the potential ways to explore the data and then be able to increase the performance by figuring out what the problems are that your implementation has along the way. This is because in the field of ML, it is not always the case that you will get the best results from the beginning, and as you need to iteratively improve the performance as you find ways of fixing the problems your implementation faces.

Finally, we also observed how the performance between both methods is comparable; thus, in the context of ML Engineering, it is necessary to think about latency, cost, accuracy, and fine-tuning requirements of the system you're making to come up with the appropriate conclusion for your system.

## Next steps for the reader:

If you'd like to get better as a machine learning engineer, one of the things to consider is that it is not sufficient to have code that just works in a notebook. Thus, I'd encourage you to think of the following questions and experiment with trying to figure out the solution to said questions:

- How can you abstract these different methods such that very little code needs to be written if you'd like to dynamically change between the different methods?
- How could you store your fine-tuned models?
- How would you host the inference method so that it is accessible as an API?
- How could you measure (and speed up) your inference time?

Thank you!
