
This note deals with state-of-the-art methodologies in prompt tuning and addresses some distinctions between different methods used in different research.

This note compares methodologies of the following papers and what they do namely:
- [TaIDPT: Texts as Images in Prompt Tuning for Multi-Label Recognition](https://arxiv.org/abs/2211.12739)
*  [L2P : Learning to Prompt for Continual Learning 2022 ](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf)
* [CPP : Steering Prototypes with Prompt-tuning for Rehearsal-free Continual Learning 2023](https://arxiv.org/abs/2303.09447)
* [CODA-P: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning 2023](https://arxiv.org/abs/2211.13218v2)
* [MLCIL: Dynamic Prompt Adjustment for Multi-Label Class-Incremental Learning 2024](https://arxiv.org/html/2501.00340v1)
* [PGP: PROMPT GRADIENT PROJECTION FOR CONTINUAL  LEARNING]()
* [NSP: Visual Prompt Tuning in Null Space for Continual Learning]()

The first paper is in a single task setting while the others are in CIL. 

CIL questions for prompt tuning: How do we perform continual learning with two core concepts - plasticity + stability? How do we generate prompts and how to create an objective function? What about the input to the models and how to optimize? 

# Summaries

# Methodologies
## Prompt Pool and Selection
Prompts - Just like Prompt Tuning (PT) in NLP where they use frozen transformer models for downstream tasks prepended to input tokens, here prompts are **learnable embedding vectors**. 

### Key - Value based (L2P)
- **L2P** - Keys learnt via a *keys to query loss* function 
![[Pasted image 20250517093628.png]]
- To choose **diverse prompts**, a penalty term added to the equation, based on a **prompt frequency table**.
- They are **task-agnostic** and keys are just selected by minimizing the input. 
### Prompt Components (CODA-P)
- **CODA-P** - Weighted summation of individual **prompt components** to calculate the final prompt.
- The **keys** to prompts are learned by using a **clustering** technique by projecting into a **latent dimensional space**.
- Each component has an **attention vector** which works like a **feature selector** in the latent space.
- Prompt component, attention and key compared with the query for obtaining **weights** for the prompt component. In essence, the contribution of each prompt component can determine the final prompt, which is determined by the similarity between the query and the key.
![[Pasted image 20250517125415.png]]
- 
- Again, **task-agnostic**.
### Contrastive Prototypical Loss (CPP)
- **CPP: CLP** technique done to combat **interference** and to cluster classes within a task.
- Interference - Classes from prior tasks overlapping with current tasks because of close embedding values in a space.;.
- Objective? Cluster together intra-task classes while avoiding interference.
- **Prototypes** - Multi-centroid embeddings for each class taken (in place of just mean distribution from class). The embeddings are taken by using **pairwise cosine sim** between each class embedding and then **spectral clustering** performed to get C no. of centroids.
- These prototypes are used for performing CPL. 
- **Negative anchors** - classes from previous tasks, **negative embeddings** - classes in current task thats not associated with a prototype, **positive embeddings** - current task, current class.
- Prototypes decoupled into **two items**: **Key** - which uses the multi-centroid embeddings value C which used for querying, and **Value** - its the task specific prompt embedding for the particular class.
![[Pasted image 20250517125627.png]]
- During learning, the distance between query vector and the Value prototype are minimized.
- **Deep Prompt** used, so they append the prompts to S layers.

###  Dual Prompting (TaIDPT)
- In a zero shot setting - text descriptions available for images in a dataset are used for generating class labels by performing a **noun filtering**. Done by stemming the nouns in the description sentences, and then matches with a **synonym table** to find appropriate class variables.
- Two prompts are learnt while maximizing the similarity between the input image and the ground truth classes - global embedding which takes the global text and image embeddings and a local embedding for fine grained features which is useful in a multiclass setting where minority classes are to be identified as well in images. 
![[Pasted image 20250522160946.png]]
![[Pasted image 20250522161241.png]]
- They use two copies of the Enc_t(r) encoder which comes from CLIP's text encoder for each prompt. Then the global features f, h for image and text embeddings are respectively calculated along with F, H for local. 
- During training, the prompt parameters are trained and during testing, the class embeddings are obtained. During training only text descriptions are used for the training, and during texting the image is mapped with the classes and hence visual features are used. They argue that just using the classes in text embeddings are enough for the training and visual features are not needed.
### Incremental Context Prompt (MLCIL)
- **MLCIL** - Uses two different prompts: **category specific** and **context specific**.
- Aim for using two prompts? To reduce class specific bias.
- Each takes word embeddings representative of each learned class and the context respectively.
- **Feature selection** performed by using image encoder and a text encoder in CLIP. 
- Three output feature vectors obtained from these models - **fx** (visual features of image x for the respective classes), **gc** (textual features flr category specific prompts), gs (context specific).
- **CFA** - Class specific region feature aggregation - image features extracted for each relevant category
![[Pasted image 20250517132707.png]]
- They aggregate the text features and image features by projecting the image features into textual feature space.


## Catastropic forgetting specific methods
### Expansion and Orthogonality (CODA-P)
- CODA-P: **Freezes** past components an only updating new components and *expanding* the set.
- To reduce **interference**, they use orthognality constraint to the Prompt, Key and Attention. Its included as a penalty term in the loss.
![[Pasted image 20250517143721.png]]
### Selective Confidence Cluster Replay (MLCIL)
- MLCIL: Clusters of positive samples are obtained and then **confidence based cluster sampling** is done.
- Purpose - It basically retrains k number of **hard samples** selected within a cluster and they are retrained. 
- These k samples are chosen by their lowest predicted probability from the text encoder.

### Orthogonal Projection (PGP & NSP)
The main idea is to avoid catastrophic forgetting by preserving learned weights, by projecting upcoming new inputs from tasks in the orthogonal direction of the previously learned tasks. Both the following papers do this at the attention level.
- **Commonalities between the two research**
    - At the attention layer, the weights of queries and keys are frozen. 
    - A combined embedding dimension is used which includes the input and prompt, and this is used to check for orthogonality.
- **PGP**
    - The combined input space/sum space is taken and the SVD gives us a **gradient projection matrix** (or V value) which can be used for the orthogonality condition.
    - Prompt tuning utilizes the **task identifier** in gradient projection.
    - Prompt gradient is computed during the backward projection by satisfying the orthogonal condition.
    - The anti-forgetting condition only uses the current prompt, input and the prompt and input for the subsequent task.
    - ![[Pasted image 20250609132904.png]]
    - ![[Pasted image 20250609132941.png]]
    - For key-query based prompt tuning, we would want to preserve the distance between learned queries and keys in the prompts from previous tasks.  This is done with the following condition - ![[Pasted image 20250609133549.png]]
    - The gradient projection matrix V can be split into two parts, the first part would be the data from previously learned tasks and the second part would be the new data. The gradient of the subsequent task would only rely on the gradient projection matrix of the new data. So they use an ε **stability-plasticity** variable to control how many values of the new data they can take and how many old. That is $V[:, 1 : εn]$, $V[:, εn : n]$.
 - **NSP**
     - 

## Parameter Efficient Fine Tuning (PEFT) Techniques


# Datasets & Models

### Requirements for dataset
- Multi-label dataset with images and text descriptions so that both image and text can be processed using CLIP's vision and language encoders.
- Classes can either be explicit or they can be derived by a methodology similar to TaIDPT.
- Continual Learning should be possible by splitting classes where in successive selective prior task classes should reappear.

### Dataset options
**MS-COCO**: 
- 80 distinct categories, 330k images, large scale dataset for object detection, segmentation and captioning tasks in computer vision.  
- *Annotations?* Bounding boxes, segmentation masks, 1.5 million captions in total.
- *Features?* Multiple object instances, diverse images.
**PASCAL VOC**:
- 20 distinct classes, 10k images, 24,640 annotated objects; small-scale but good for multi-label scenarios.
- Person, animal, vehicle, indoor objects are the category of the 20 classes.
- *Annotations?* Bounding boxes, object class label for each object.
**NUS-WIDE**:
- 270k images, 81 distinct classes with 5k total tags associated from Flickr.
- *Annotations?*: Apart from bounding boxes, object class labels, also provides low level features like color histogram, color correlogram, edge direction histogram, wavelet texture, etc. 

### Model options
**CLIP**:
- Text encoder and image encoder can be used separately to create a joint embedding with image descriptions and images in a joint embedding space.
- Two prompts can be used in a multi-class scenario, one for finding localized features label specific and another for global setting which would include context embedding.
- To address catastrophic forgetting, catefgory specific features from old tasks can be sampled (using ROI like in SCCR), and then clustering can be performed to get clusters of features for each category. 
- Another approach can be to use something similar to Jia et al.[^1], where they create a "disentangled representation" of each label. Label-related spatial regions are identified using *learnable semantic queries*. The semantic queries aims to adhere to all the differences in semantic consistencies for the label in different images. Ex.: baby, woman, man all should be having *person* label. Binary classifiers for each label is formed and label features calculayed using the aforementioned algorithm are sent to each label's classifier for identifying the presence of the label. Although this was implemented using ResNet101 and Swin Transformer, maybe something equivalent can be performed with a VL model.


[^1]: J. Jia, F. He, N. Gao, X. Chen, and K. Huang, ‘‘Learning disentangled label representations for multi-label classification,’’ 2022, arXiv:2212.01461.
