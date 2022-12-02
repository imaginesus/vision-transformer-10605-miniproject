---
layout: post
title: Vision Transformers for Image Recognition at Scale
authors: Sushanth Reddy (syellare), Maitheli Biswas (maithelb), Annie Johnson (anniej)
---

Contents of the blog...

1. [Transformers in Image Recognition](#transformers-in-image-recognition)
2. [Comparing CNNs and Transformers](#comparing-cnns-and-transformers)
3. [Vision Transformer Model Design](#visiontransformermodeldesign)
4. [Comparing the Vision Transformer with ResNet](#comparingthevisiontransformerwithresNet)
5. [Vision Transformer Variants and Experimental Results](#latex)
6. [Scaling the Vision Transformer](#references)
7. [Conclusion](#references)
8. [References](#references)

## Transformers in Image Recognition

The idea of transformers has been very well adapted in the world of natural language processing (NLP) owing to their ability to learn relationships between different positions of sequential elements with great accuracy. Self attention architecture in NLP allows for finding meaningful representations of all words within the same sentence at once making it a simple matrix calculation and hence allowing the scope for parallelism leading to computational efficiency and also scalability. 

The computer vision (CV) world has tried to imbibe transformers architecture but usually has been out bid by the adoption of convolutional architectures mainly due to issues with accuracy and scalability of transformers. When trained on mid-sized datasets, Transformers tend to underperform compared to CNNs but when given larger datasets of images, they tend to outperform CNNs. Even in NLP, Transformers are first applied to a large text corpus and then fine-tuned for the current task at hand. 

So why did the need even arise for applying transformers to CV-based problems? Convolutional Neural networks (CNNs) are really powerful when it comes to extracting features but need computationally intensive large filters to find relative positions of different features which are far apart which is where Transformers shine, both in terms of accuracy and speed. We explore Vision Transformers (ViT) in this blog which are one of the various adaptations of Transformers for CV purposes. ViTs use self-attention on sequences of extracted patches from images so that each pixel can try to relate to every other pixel in a given patch on a large pre-training dataset. 


## Comparing CNNs and Transformers

Before we dive further into ViTs, here is a quick summary of the key differences between CNNs and Transformers...

| ![CNN](https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/cnn.png) | ![Transformer](https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/transformer.png) |
|:--:| :--:| 
| *Figure 1: CNN architecture* | *Figure 2: Transformer architecture* |

<table>
  <thead>
    <tr>
      <th> </th>
      <th>Convolutional Neural Networks (CNNs)</th>
      <th>Transformers</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>Architecture</td>
      <td>CNNs usually have
        <ul>
        <li>convolutional layer which extracts the feature map from an image</li>
        <li>pooling layer reduces the size of convolved feature map</li>
        <li>fully connected layer which perform the classification task</li>
        </ul>
      </td>
      <td>Transformers consist of
        <ul>
        <li>encoders which are to create embeddings from the input sequence</li>
        <li>decoders which take in previous outputs (start-of-sequence token and right-shifted output sequence by one position) and the embeddings from the encoder to produce the next element in the output sequence</li>
        </ul>
      </td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>Inductive Biases</td>
      <td>CNNs usually have locality and weight sharing based inductive biases. Translation invariance with pooling layers and translation equivariance without pooling layers are also few other inductive biases that are built into the models.</td>
      <td>Transformers tend to not have strong inductive biases and hence are more flexible but at the same time, need a huge amount of data to find the optimum.</td>
    </tr>
    <tr>
      <td>Performance & Accuracy</td>
      <td>CNNs have the ability to localize features using filters to extract important features and edges. The inductive biases mentioned above help CNNs generalize from observed examples and hence have higher accuracy with medium-sized datasets.</td>
      <td>Transformers tend to not perform well when not given enough data. But if provided with sufficient data in the pre-training phase, Transformers tend to have higher accuracy than CNNs.</td>
    </tr>
    <tr>
      <td rowspan="2">Computational Complexity</td>
      <td>Complexity per convolutional layer: O(knd<sup>2</sup>)</td>
      <td>Complexity per self-attention layer: O(n<sup>2</sup>d)</td>
    </tr>
    <tr>
    <td colspan="2"><em>where n is sequence length, d is representation dimension, k is kernel size</em></td>
    </tr>
    <tr>
      <td>Path length between dependencies</td>
      <td>A single convolutional layer does not connect all pairs of input and output and hence we need a number of stack of these layers to increase the path length </td>
      <td>Self-attention layers connect all positions available in the input sequence and hence cover even the long-range relationships with lower computational requirements.</td>
    </tr>

  </tbody>
</table>

## Vision Transformer Model Design

<p align="center">
  <img src="https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/vit_model.png" alt="ViT model"/>
  <figcaption align="center">Figure 3 - ViT model architecture</figcaption>
</p>

text

## Comparing the Vision Transformer with ResNet

Now let us analyze the model performance of the Vision Transformer (ViT) with a classic large-scale image recognition model like ResNet as the benchmark. The article compares these models based on multiple criteria like- classification accuracies on multiple datasets, size of datasets required for pre-training, and the pre-training compute time for the models.

As noted previously, Transformers lack inductive biases inherent to CNNs. Thus to trump inductive bias, we train the Vision Transformer (ViT) on larger datasets (14M-300M images). After model pre-training, we transfer the models to classify several benchmark datasets, as shown in table<>. It has been observed that Vision Transformers (ViT) outperform ResNets with the same computational budget.

Another way to induce a sliver of inductive bias is to use CNN [1] feature maps as Inputs to our Vision Transformer instead of image patches. The input sequence of the Transformer is created by flattening the CNN feature maps and applying the input patch embedding and positional embeddings as described in the previous section. The authors call this a "Hybrid ViT".

## Vision Transformer Variants and Experimental Results

For the Vision Transformer, we experiment with three model variations, as detailed in Table 1. We use brief notation to indicate the model size and the input patch size: ViT-L/16 refers to our Vision Transformer model "Large" variant with a 16×16 input image patch size. Since Transformer's sequence length is inversely proportional to the square of the patch size, thus models with smaller patch sizes are computationally more expensive. The performance of each of these models is compared to our baseline "ResNet (BiT)" [2] that performs supervised transfer learning with large ResNets. 

<p align="center">
  <img src="https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/vit_variants_table.png" alt="ViT variants"/>
  <figcaption align="center">Table 1 - ViT model variants</figcaption>
</p>

The table below shows the comparative results of the Vision Transformer variants with the "ResNet BiT" on multiple benchmark datasets. The ViT-L/16 model trained on the JFT dataset outperforms the baseline "ResNet (BiT)" model on all tasks while requiring substantially fewer computation resources (in TPU v3 core days).

<p align="center">
  <img src="https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/model_performance_table.png" alt="model performance table"/>
  <figcaption align="center">Table 2 - ViT variants model performance compared to ResNet(BiT)</figcaption>
</p>

Model Pre-training Requirements

Based on the requirement of pre-training the Vision Transformer on large datasets before transferring the model, It begs the question of how crucial is the pre-training dataset size. To Examine this, two sets of experiments are performed.

Experiment 1: Pre-Training ViT models on increasing dataset size.

Below are the results of training ViT variants and ResNet on ImageNet, ImageNet-21k, and JFT datasets. We see that larger variants of ViT (ViT-Large and ViT-Huge) underperform the ViT-Base variant on the smaller pre-train dataset. But this changes as the pre-training dataset size increases where larger ViT variants perform better than ViT-Base, and also outperform ResNet by pre-training on JFT 300M dataset. 

<p align="center">
  <img src="https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/dataset_vs_models.png" alt="dataset vs models"/>
  <figcaption align="center">Figure 4 - Model performance vs pre-training dataset</figcaption>
</p>

Experiment 2: Pre-Training on subsets of the full JFT 300M images dataset

Below are the results of training the ViT variants using the same hyperparameters on multiple subsets of the JFT 300M images dataset. Using the same hyperparameters helps us judge the intrinsic model properties of each ViT variant. We observe that ResNets perform well on smaller subsets of the dataset due to the inductive biases of CNNs. But as the dataset size increases, learning relevant patterns from the data trumps the lack of inductive biases in Transformers, and the ViT models perform better than ResNet models. 

<p align="center">
  <img src="https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/JFT_subsamples_vs_models.png" alt="JFT vs models"/>
  <figcaption align="center">Figure 5 - Model performance vs JFT pre-training subset dataset size</figcaption>
</p>

## Scaling the Vision Transformer

To understand how the Vision Transformer model compares with ResNet when controlled for pre-training compute time, we assess the performance vs. computation cost of several variations of each model. The Authors also throw the "Hybrid ViT" model into the mix to analyze its performance compared to ViT and ResNet.

The figure below shows the transfer performance time vs. total pre-training compute time of the models under study. As observed from table<>, The ViT models outperform the ResNet models, with the Hybrid ViT models showing a slight outperformance compared to the traditional ViT models. This is a surprising find as we expect that inductive bias from CNN feature maps would assist the ViT model in better training and classification by our Hybrid ViT variant.

<p align="center">
  <img src="https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/scaling_study.png" alt="scaling study"/>
  <figcaption align="center">Figure 6 - ViT vs Hybrid ViT vs ResNet: Accuracy vs Pre-Training time</figcaption>
</p>

## Conclusion

After conducting the experiments, the authors inspect the Vision Transformer to understand how the model processes image data and why it matches or outperforms state of the art CNNs. Similar in function to the lowermost layers of CNNs, the multi-headed self-attention allows the ViT to leverage a highly localized attention which helps the model attend to the semantically relevant regions of the image for classification. The attention distance metric is used to investigate to what degree the ViT integrates information from an image. The attention distance is the average distance in the image space across which information is integrated. From figure 4 it is clear that attention distance increases with network depth which implies that with depth the model attends widely across relevant tokens rather than attending to all regions of the image. The authors also performed a masked patch prediction task similar to the masked language modeling task by corrupting 50% of the patch embeddings. Their ViT-B/16 model with self-supervised pre-training attained an accuracy of 79.9% accuracy on ImageNet which performs 2% better than training from scratch, but still 4% behind supervised pre-training. The authors have left contrastive pretraining to future work. 

<p align="center">
  <img src="https://raw.githubusercontent.com/imaginesus/10605_mini_project_images/main/depth_vs_attention.png" alt="depth vs attention"/>
  <figcaption align="center">Figure 7 - Plot of Mean attention distance versus Network depth of the ViT-L/16 model.</figcaption>
</p>

This research work presents a simple yet scalable strategy to perform image classification that supersedes conventional CNN architectures when combined with pre-training on large datasets. However, the authors also mention that this architecture could be employed in other computer vision applications as well. \ref{khan2022transformers} is a survey paper that discusses many other works that extend the ViT to not only recognition tasks like image classification, object detection, action recognition, and segmentation, but also to generative modeling, multi-modal tasks like visual question answering, video processing, low-level vision and 3D analysis. The study also mentioned transformers like Linformer introduced in \ref{wang2020linformer} and Reformer presented by \ref{kitaev2020reformer} that reduced the complexity of self-attention from O(n2) to O(n) and O(n log(n)) respectively. This study concluded that for the different tasks, scaling up in terms of compute, model size and quantity of training data improves performance of the vision transformer. Overall, this research work has had a significant impact on the domain of computer vision as well as developing compute efficient models for large scale training of images.

## References