This note would extend on the state-of-the-art methods in multi-label image classification tasks. This would also touch up on some multi-label class incremental learning although, while this note was written, the work on this topic is rather limited. The main papers this note uses for its research are listed below. A list of full references can be found at the bottom of the page.

- [MLCIL: Dynamic Prompt Adjustment for Multi-Label Class-Incremental Learning 2024](https://arxiv.org/html/2501.00340v1)
- [REVIEW1: Multi-Label Lifelong Machine Learning: A Scoping Review of Algorithms, Techniques, and Applications](http://ouci.dntb.gov.ua/en/works/4badWvB7/)


# Existing Techniques for multi label CIL (REVIEW1)

## Regularization Techniques
Minimizes catastrophic forgetting by preventing model overfitting.
**Elastic Weight Consolidation[^1]** - One of the most naive ways to do it; slow down learning by imposing a *quadratic penalty* on discrepancies found between previously learned parameters and the current task's relevant weights.
**Learning without Forgetting (LwF)**[^2]: Uses CNNs; Uses *Knowledge Distillation* to calculate parameters of current task from the losses of previous tasks, in order to keep them similar and not vary too much. It optimizes the parameters of both the shared and current task at the same time in this process.![[Pasted image 20250520140845.png]]
## Rehearsal
Selective reintroduction of past data samples.
### Exemplar Rehearsal
Involves storing a buffer of selected past data samples representative of previously observed distribution.
**[Examples]**
**Gradient Episodic Replay[^3]**  Does the above (but memory and computational costs to be considered)
**Partition Reservoir Sampling[^9]**: Modified version of *Reservoir Sampling Technique* - maintains a reservoir of elements that represent a true random sample as the data stream introduces new elements. Each element has an equal probability of getting replaces at each time step by a new element that an better help represent the sample batch. Label frequencies are used to identify majority classes. 
**Optimizing Class Distribution in Memory (OCDM)[^10]**: Memory-based technique dynamically maintains a representative distribution of classes in memory (replay buffer) which is a small amount of previously seen data. Memory update mechanism is treated like an optimization problem, where the memory is first filled with observed mini-batches of data and at each time step data in the memory is replaced with a new distribution closest to a target distribution. (sorry if that was a long running sentence)
### Generative Replay 
Uses generative models like GANs and VAEs for creating synthetic data samples and they are then combined with the current task's data during training.
**[Examples]**
**Deep Generative Replay[^4]** uses two models - one generative model to create the synthetic data and the other is a task solver. Reduced working memory requirements since you can delete the old training samples.
## Parameter Isolation
Modifies underlying architecture by either updating a fixed number of parameters in the model or by dynamically growing the architecture by either adding new neurons or modules or layers upon each new task. The fixed number of parameters methodology is not worth noting about since it could lead to *overparameterization*.
**[Examples]**
**Progressive Neural Networks[^5]**: A pool of pre-trained models is retained, one for each learned task.  Lateral connections between existing task and learned tasks are learned to facilitate knowledge transfer.

## Clustering Techniques
Generating clusters from data points to find target class by finding the closest centroid(s) between the clusters. Conventional kNN eploys a criterion like *majority voting* to assign a class. Now how do we extend that to multi-class scenarios? Are there any other ways to cluster the data?
**[Examples]**
**Roseberry et al. 2021[^6]**: Introduce a self-adopting algorithm to address *multi-label drifting data streams*. In other words, when statistical properties in the data change from one task to another. They perform kNN clustering for multi-class setting by using the *frequency* of each label's occurrence to the closest neighbours in the training set.
**Masuyama et al.[^7]**: They utilize a combination of Bayesian technique and a cognitive neural theory called *[Adaptive Resonance Theory](https://www.geeksforgeeks.org/adaptive-resonance-theory-art/)*[^8] for tracking label probabilities and for simultaneously solving the stability-plasticity problem respectively. They cluster instances with similar label patterns and identify underlying relations in data, hence reducing the size of the output space learnt. Their algorithm works well with more and more labels.

## Binary Classifiers
In this technique, each label is treated independently and it neglects the co-existence of other labels. It decomposes multi label classification into separate binary classification tasks for each label. But disregarding these label relations can cause sub-optimal performances. This is the opposite strategy than using pairwise relations for finding relevant and irrelevant relations. But having high order relations modeled especially in a case where the number of labels are high, then learning from such a vast output space wouldn't be needed when using binary classifiers. 
**[When to use it]**
When you have either a high number of labels and/or label dependencies are low for the dataset you're working on.
**[Examples]**
**Compact, Picking and Growing (CPG)[^11]**: Continual Learning for *defect detection* in manufacturing pipelines. Their model learns new defects without retraining prev data. Each task is formulated as a binary classification problem for each defect type. CPG identifies crucial weights in the deep neural network learned from prior tasks, and the network's size increases.

## Hashing
Binary codes are learnt from images/audio. Efficiency in storage and searching for large-scale retrieval of multi-label images/audio. 
**[Examples]**
**Continual Deep Semantic Hashing (CDSH)[^12]**: Two hashing networks - one for hashing semantics of data into semantic codes and other for mapping images to corresponding semantic codes. It uses *empirically verified loss* and a special regularization design to ensure old labels remain unchanged. Efficient at that.

## Features in Labels
**[Examples]**
**Disentangled Label Feature Learning (DLFL)[^13]** 
*Disentangled representation* of each label is created by using a feature disentanglement module called *One-Specific-Feature-for-One-Label* mechanism. *Semantic Spatial Cross Attention (SSCA)* localizes label related spatial regions into corresponding label features for the disentanglement.
**Feature Classification and Probing Unknown (HCP)**[^14] 
Feature purification module aims to create unique representations for each class. *Why?* To avoid feature aliasing between tasks - and to focus only on target features of a class. 
*Patch tokens* - Global features are reshaped to a certain dimension to be combined with class features.
This whole module contains multi-head attention modules and can get the spatial information for global feature tokens along with contextual relationships between class embeddings.
*Stability classifier* predicts old logits of historical classes by passing the class features as input into their classifier. There's parallely a *Plasticity* classifier that predicts logits of newer classes. 
*Recall Enhancement* - To check which classes the model might've forgotten, class probabilities are obtained by calculating the mean and variance of confidence for each class. *Confidence forgetting* is defined as the difference between maximum confidence gained through past tasks and the confidence after finishing current task.
*Probing Unknown Knowledge*: Aim is to reserve embedding space for future classes and enhance model's forward compatibility. They do it by creating synthetic features. They do this by computing features of classes that are absent in an image by interpolating randomly using beta distributions and feeding these features into the classifier. The goal? To distance known features from synthetic features in the embedding space. 


# References
[^1]: J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, and A. Grabska-Barwinska, ‘‘Overcoming catastrophic forgetting in neural networks,’’ Proc. Nat. Acad. Sci. USA, vol. 114, no. 13, pp. 3521–3526, 2017.

[^2]: Z. Li and D. Hoiem, ‘‘Learning without forgetting,’’ IEEE Trans. Pattern Anal. Mach. Intell., vol. 40, no. 12, pp. 2935–2947, Dec. 2018.

[^3]: D. Lopez-Paz and M. Ranzato, ‘‘Gradient episodic memory for continual learning,’’ in Proc. Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 1–10.

[^4]: H. Shin, J. K. Lee, J. Kim, and J. Kim, ‘‘Continual learning with deep generative replay,’’ in Proc. Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 1–10.

[^5]: A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, ‘‘Progressive neural networks,’’ 2016, arXiv:1606.04671.

[^6]: M. Roseberry, B. Krawczyk, Y. Djenouri, and A. Cano, ‘‘Self-adjusting K nearest neighbors for continual learning from multi-label drifting data streams,’’ Neurocomputing, vol. 442, pp. 10–25, Jun. 2021.

[^7]: N. Masuyama, Y. Nojima, C. K. Loo, and H. Ishibuchi, ‘‘Multi-label classification via adaptive resonance theory-based clustering,’’ IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 7, pp. 8696–8712, Jul. 2023.

[^8]: G. A. Carpenter and S. Grossberg, ‘‘Adaptive resonance theory,’’ Dept. Cogn. Neural Syst., Center Adapt. Syst., Boston Univ., Boston, MA, USA, Tech. Rep. CAS/CNS TR-98-029, 2010.

[^9]: C. D. Kim, J. Jeong, and G. Kim, ‘‘Imbalanced continual learning with partitioning reservoir sampling,’’ in Computer Vision—ECCV. Glasgow, U.K.: Springer, 2020, pp. 411–428.

[^10]: Y.-S. Liang and W.-J. Li, ‘‘Optimizing class distribution in memory for multi-label online continual learning,’’ 2022, arXiv:2209.11469.

[^11]: C.-H. Chen, C.-H. Tu, J.-D. Li, and C.-S. Chen, ‘‘Defect detection using deep lifelong learning,’’ in Proc. IEEE 19th Int. Conf. Ind. Informat. (INDIN), Jul. 2021, pp. 1–6.

[^12]: G. Song, K. Huang, H. Su, F. Song, and M. Yang, ‘‘Deep continual hashing for real-world multi-label image retrieval,’’ Comput. Vis. Image Understand., vol. 234, Sep. 2023, Art. no. 103742.

[^13]: J. Jia, F. He, N. Gao, X. Chen, and K. Huang, ‘‘Learning disentangled label representations for multi-label classification,’’ 2022, arXiv:2212.01461.

[^14]: Zhang, Aoting, et al. "Specifying What You Know or Not for Multi-Label Class-Incremental Learning." _Proceedings of the AAAI Conference on Artificial Intelligence_. Vol. 39. No. 21. 2025.
