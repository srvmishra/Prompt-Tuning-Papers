# Prompt-Tuning-Papers
This repository contains recent papers for prompt tuning

## Continual Learning
Multi class setup - One sample has only one label. (1, 4 -> 3, 5, 6, 7 -> 2)

Usually there are no common classes between two consecutive tasks. So tasks are disjoint both sample wise and class wise.

1. [Learning to Prompt for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf)
2. [Choice of PEFT Technique in Continual Learning: Prompt Tuning is Not All You Need](https://arxiv.org/abs/2406.03216)
3. [Prompt Customization for Continual Learning](https://arxiv.org/html/2404.18060v1)
4. [Steering Prototypes with Prompt Tuning for Rehearsal-Free Continual Learning](https://openaccess.thecvf.com/content/WACV2024/papers/Li_Steering_Prototypes_With_Prompt-Tuning_for_Rehearsal-Free_Continual_Learning_WACV_2024_paper.pdf)
5. [Prompt Gradient Projection for Continual Learning](https://openreview.net/forum?id=EH2O3h7sBI)
6. [Visual Prompt Tuning in Null Space for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f06be0008bc568c88d76206aa17954f-Abstract-Conference.html)
7. [Dynamically Anchored Prompting for Task-Imbalanced Continual Learning](https://www.ijcai.org/proceedings/2024/0456.pdf)

## Multi Label Learning
One sample has multiple labels. Different samples can have some common labels between them.

1. [Texts as Images in Prompt Tuning for Multi-Label Image Recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Texts_as_Images_in_Prompt_Tuning_for_Multi-Label_Image_Recognition_CVPR_2023_paper.pdf)

## Continual Multi Label Learning
Since one sample has many labels, even if we divide the samples into different tasks, there might still be common classes between consecutive tasks. Tasks are disjoint only sample wise. So, a sample from task 2 might contain classes from task 1. From the dataset for task 2, we only have access to task 2 labels, and not task 1 labels. So, first we must use the model trained on task 1 to predict these missing labels on task 2 data and then use these predicted task 1 labels along with the ground truth task 2 labels to train the model on task 2 data. This usually causes problems. 

1. [Dynamic Prompt Adjustment for Multi-Label Class Incremental Learning](https://arxiv.org/html/2501.00340v1)

## Other References

1. For continual multi-label learning data splits and evaluation schemes only: [Knowledge Restore and Transfer for Multi-Label Class Incremental Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Dong_Knowledge_Restore_and_Transfer_for_Multi-Label_Class-Incremental_Learning_ICCV_2023_paper.pdf)
2. Labeling the older classes from previous tasks in current images correctly before using these samples to train the new model - [Confidence Self-Calibration for Multi-Label Class-Incremental Learning](https://arxiv.org/abs/2403.12559)
3. Stability vs Plasticity trade off - can be used to inform what to replay - [Specifying What You Know or Not for Multi-Label Class-Incremental Learning](https://ojs.aaai.org/index.php/AAAI/article/view/34390/36545)
