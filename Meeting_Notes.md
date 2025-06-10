# Meeting Notes
This note discusses the weekly meeting MOMs.

## 17.05.25

**Main Idea** : Prompt Tuning by having one prompt per image as opposed to one prompt per class or task. Prompts learnable components that try to minimize the distance between prompt and image and can be looked up during inference.

### **Paper Discussion**
1. [Learning to Prompt for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf)
2. [Steering Prototypes with Prompt Tuning for Rehearsal-Free Continual Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Li_Steering_Prototypes_With_Prompt-Tuning_for_Rehearsal-Free_Continual_Learning_WACV_2024_paper.pdf)
3. [Dynamic Prompt Adjustment for Multi-Label Class Incremental Learning](https://arxiv.org/html/2501.00340v1)
4. [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](https://arxiv.org/abs/2211.13218v2)
### **Discussion points**
- Unlike paper[2], we want prompts to be generic. Contrastive methods would not work in a multi-label scenario.
- Number of prompts cannot be fixed like in paper[1]. Pre-trained frozen LLMs can be used.

## 24.05.25

### **Paper Discussion**
1. [Texts as Images in Prompt Tuning for Multi-Label Image Recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Texts_as_Images_in_Prompt_Tuning_for_Multi-Label_Image_Recognition_CVPR_2023_paper.pdf)
2. [Learning Disentangled Label Representations for Multi-label Classification](https://arxiv.org/abs/2212.01461v1)
3. [Prompt Customization for Continual Learning](https://arxiv.org/html/2404.18060v1)
4. [Learning disentangled label representations for multi-label classification](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://arxiv.org/abs/2212.01461&ved=2ahUKEwjE06TWzeaNAxX1Q2cHHbbdGbgQFnoECAkQAQ&usg=AOvVaw29G4Fqlt3-DU4y-J7KmenN)

### **Discussion points**
- We want CLIP driven multi-label image setup mostly. We can pass text-based prompts to the LLM and the embedding space would be shared between the prompts and images. 
- We need to address both catastrophic forgetting as well as class imbalance in data.
- Catastrophic forgetting can be addressed by using RL: for keeping track of forgotten classes. A reward model can be used to inform processes which is replaying data for us. Replay process can give us textual prompts. Later, it can also be used to retrieve images which will contain objects of a particular class.

## 31.05.25

### **Paper Discussion**
1. [Prompt Gradient Projection for Continual Learning](https://openreview.net/forum?id=EH2O3h7sBI)
2. [Dynamically Anchored Prompting for Task-Imbalanced Continual Learning](https://www.ijcai.org/proceedings/2024/0456.pdf)
3. [Confidence Self-Calibration for Multi-Label Class-Incremental Learning](https://arxiv.org/abs/2403.12559)
4. [Specifying What You Know or Not for Multi-Label Class-Incremental Learning](https://ojs.aaai.org/index.php/AAAI/article/view/34390/36545)

### **Discussion points**
- Catastrophic forgetting can be focussed on by injecting prompts from previous tasks - augmentation prompting.
- Reward function can be used to keep track of this misallignment for previous task prompts.
- Class imbalance can probably be addressed based on paper[2] by introducing a lambda operator to balance classes.
- KRT does forgetting calculation at the attention level. instead of doing it in a heuristic way, this might be more effective for identifying forgotten classes and can be adapted to our model.
- Orthognality condition from paper[1] might come to use since it's directly in the attention layer and straightforward.
- Codebase for binary classification problem and dropout layer in backward pass will be sent soon.

## 09.06.25

### **Paper Discussion**
- [Visual Prompt Tuning in Null Space for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f06be0008bc568c88d76206aa17954f-Abstract-Conference.html)
- [Choice of PEFT Technique in Continual Learning: Prompt Tuning is Not All You Need](https://arxiv.org/abs/2406.03216)
- [A Unified Continual Learning Framework with General Parameter-Efficient Tuning](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_A_Unified_Continual_Learning_Framework_with_General_Parameter-Efficient_Tuning_ICCV_2023_paper.html)

### **Discussion points**