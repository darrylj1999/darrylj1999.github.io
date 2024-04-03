Luting Wang, lwang797@gatech.edu, Darryl Jacob, djacob30@gatech.edu 

Music Genre Classification

Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

# Dataset Description:
Our dataset comprises 3-second excerpts from songs spanning 10 distinct genres. Each data point is characterized by 60 features, encompassing key audio metrics such as the root mean square value, spectral centroid value, and zero-crossing rate. These are all metrics used within the music industry to quantify sound. Each of the 10 genres is represented by 100 songs, and each song is further divided into 10 separate 3-second excerpts. This division ensures a comprehensive and varied representation of each genre, contributing to the diversity of our dataset.

# Lit Review:
https://ieeexplore.ieee.org/document/9585937
Previous papers have shown various approaches to classifying and predicting genre of songs from musical features. For example, Leartpantulak et al. have shown a stacking ensemble method which involves two classifiers, a base (K-Nearest Neighbors (k-NN), Decision Tree (DT), Random Forest, Support Vector Machines (SVM); and Naïve Bayes) which classifies and is then sent to a meta classifier [5]. Other groups, such as Ghosal et al. employed convolutional neural networks to extract features and utilized LSTM autoencoders to capture temporal dynamics when analyzing time series data [6]. We found that Gated Recurrent Units outperformed other models at the generic audio and audio scene classification tasks [7]. In fact, more layers (ex. Convolutions) can be incorporated to extend superior performance with multimodal data [8]. 

# Problem Definition
The problem at hand revolves around predicting the music genre for different songs given only audio characteristics. It can be very labor-intensive to manually match songs to a certain genre, and thousands of new songs are being released everyday. Our goal is to automate this process by developing machine learning models capable of classifying music into genres using quantitative features extracted from audio signals.

There are a couple reasons that we are motivated to address this problem. First, the sheer volume and diversity of music available today make manual genre classification a daunting task. An automated system would expedite the categorization process, aiding streaming platforms in organizing music more efficiently. 

Second, a genre prediction model contributes to enhancing user experiences in music recommendation systems. By understanding the underlying features that define different genres, personalized playlists and recommendations can be tailored to individual preferences. 

# Methods
Data Preprocessing:
- Normalization (Min-max scaling or normal scaling)
  - For better training
- PCA (unsupervised)
  - For feature reduction
- Mel-Frequency Cepstral Coefficients (MFCC) extraction [2]
  - Simulates human hearing when listening to music [2]

To aid our classification for more advanced models, we leveraged a “One Hot Encoding” system for our target, the song’s genre. Hence, we turn a discrete target with ten different values into ten targets which can take values between zero and one. We train models to incorporate multiple outputs and then choose between outputs given output type. For example, with the Gated Recurrent Unit, if we use continuous targets, we use the softmax function to scale 10 output scores to a probability distribution over 10 genres. If there is a tie between two genres, we break it by considering the prior distribution over genres. In other words, if the original dataset contains more of a certain genre, that will be the model’s predicted output when tied with a less prevalent genre. 

Models:
- Random Forest [2], [3]
  - Shown to have good accuracy (83%) by previous researchers [3]
- SVM [2], [3]
  - Shown to have decent accuracy (78%) by previous researchers [3]
- BiLSTM [2]
  - Current state of the art with this dataset, boasting the highest accuracies and F1 score (both about 94%) [2]

A Gated Recurrent Unit (GRU) provides a means to inject inductive bias into the more general Long Short Term Memory model. In this case, we maintain a variable length input, a hidden state and a cell memory. We selected the GRU because of its demonstrated task aptitude and accommodation for variable length input for significant memory savings because we do not need to store as many weights as the length of the sample. The cell memory incorporates long term information about the input by selectively forgetting and adding previous memories with new parts of the input. The hidden state is a learned mapping from the current cell memory to the target. 

The learned parameters are weight vectors used in the inner product with the concatenated current hidden and input states. The forget, input, gate and output gates are combinations of an activation function (hyperbolic tangent and sigmoid) with linear transformations of the current hidden and input states. Given how closely related this model is to conventional Artificial Neural Networks, and the differentiability of the gate functions, we train our GRU via Batch Stochastic Gradient Descent.

For this project, our inputs are 3 second excerpts from possible songs. We iterate over each frame of the input with our GRU and output the hidden state at the last frame as the probability of the track belonging to a specific genre. By batch multiplying matrices, we can treat the weights as a linear operator with a three dimensional output which gives us the relevant multiclass probability distribution with as little training time as possible. Our loss function is the Cross Entropy between the true and predicted target distributions.

# Results and Discussion

## Random Forest
We used accuracy as well as F1 score to measure the success of the random forest model. Before implementing the model, min-max scaling was implemented on all the numeric variables. This normalizes the data to ensure that data at different scales contribute equally to the model fitting. After performing the normalization, the model was trained on an 80/20 train test split. An accuracy of 0.871 and an F1 score of 0.870 was achieved by our initial random forest model. The F1 score across each genre is shown below. 


Comparable F1 scores were achieved across the different genres, with classical performing the best. Another feature of the random forest model is the calculation of feature importance. This tells us which features are most important in the model for differentiating between the genres.


Perceptron variance was the most important feature.

Future steps:

We need to implement feature selection to ensure that all features included in the model are contributing to the predictions. We can also implement hyperparameter tuning to improve our model.

Next Steps:

## SVM
### Quantitative metrics:
Accuracy: Calculate the accuracy of the SVM model on the test dataset.
Precision and Recall: Compute precision and recall scores to understand the model's performance in binary classification tasks.
F1 Score: Calculate the F1 score to balance precision and recall.

### Next Steps:

Visualizations: Plot decision boundaries for different SVM kernels (linear, polynomial, radial basis function) to illustrate how they separate classes

Analysis of Algorithm/Model: Comparison with Other Models


## BiLSTM
### Quantitative Metrics:

As accuracy and F1 score are the metrics that are often used to benchmark models in this domain, we use accuracy and F1 score to evaluate the BiLSTM.

The best results obtained were an accuracy of 0.8600 and F1 score of 0.8539. These results were obtained by normalizing data and using a hidden layer size of 256.

### Analysis of Algorithm/Model:

Taking inspiration from [2], we make the model consist of a single bidirectional LSTM layer followed by layer normalization before a final linear layer.

One thing to note is that the LSTM is a sequential model. However, the data is organized in such a way that each instance is a 3 second section of a 30 second song file. Therefore, we preprocess the data in such a way that we combine each 3 second section of each song into a sequence of 10 vectors that represent 3 seconds of the 30 second song. Therefore, each instance now becomes a sequence of vectors that make up a single song. Then, we split the dataset into train/test datasets and use mini-batches of 64 songs.

For the optimizer, we use Adam, as done by [2] as well.

Training the model at this point gives us a pretty bad training loop:

![128](https://github.com/darrylj1999/darrylj1999.github.io/assets/33488019/ff6c82b0-a3c5-4fb6-995c-2bfad25ab551)

Looking at the data more closely, we can notice that the ranges of data vary vastly, some features being at most 1, while others being at most a few thousand. To combat this, we normalize the data to reduce this noise, but also for faster training and the reduction of bias on certain features.

![norm128](https://github.com/darrylj1999/darrylj1999.github.io/assets/33488019/e0b3c444-6c31-41ce-9d9a-a68cc3ff70b3)

We now achieve very good test results right off the bat, with an accuracy of 0.84 and F1 score of 0.8346. However, we’re clearly overfitting.

One thought was that perhaps the model is too complicated. One idea was to reduce the bidirectional LSTM to a normal LSTM.

![norm128uni](https://github.com/darrylj1999/darrylj1999.github.io/assets/33488019/2a158f6b-f0ea-4ae7-9b52-ca50d102be6f)

However, this leads to similar results. This particular run achieved an accuracy of 0.8450 and F1 score of 0.8380.

Another way to combat overfitting is to reduce the number of features. The filenames, length, and tempo features were already removed due to being mostly invariant or not helpful features, thus leading to a total of 56 features that are fed into the model. One way to perform feature reduction is to utilize PCA. However, PCA tended to achieve slightly worse accuracy and F1 score.

![norm128unipca0 85](https://github.com/darrylj1999/darrylj1999.github.io/assets/33488019/e4046c33-1b3b-4714-a0c0-9e32c2adc0df)

This particular run utilized a retained variance of 85% made up of 23 principal components and had an accuracy of 0.7800 and a F1 score of 0.7743.

Yet another way to combat overfitting is to introduce regularization in the form of dropout. However, due to its random nature, dropout varied in its effectiveness, making the accuracy and F1 score better at times while worse in others.

![norm128unidrop0 2](https://github.com/darrylj1999/darrylj1999.github.io/assets/33488019/d978c992-3617-47da-a3e4-11b645e50226)

This particular run reached a peak accuracy of 0.8600 and F1 score of 0.8539.

In general, the confusion matrix looks like this:

![confusion_matrix](https://github.com/darrylj1999/darrylj1999.github.io/assets/33488019/7208f7c6-4065-46cc-928f-7081257a9973)

 ```
[[21  1  0  0  0  1  1  0  2  0]
 [ 0 19  0  0  0  0  0  0  0  0]
 [ 1  0 21  0  0  0  0  1  2  1]
 [ 0  0  0 14  2  0  1  0  0  1]
 [ 0  0  0  1 18  1  1  0  1  0]
 [ 0  3  1  0  0 18  0  0  0  0]
 [ 0  0  0  0  0  0 16  0  0  2]
 [ 0  0  0  0  0  0  0 16  2  0]
 [ 1  0  0  1  0  0  0  1 10  0]
 [ 0  0  1  2  0  0  0  0  2 13]]
              precision    recall  f1-score   support

         0.0      0.870     0.769     0.816        26
         1.0      0.947     0.947     0.947        19
         2.0      0.955     0.808     0.875        26
         3.0      0.812     0.722     0.765        18
         4.0      0.882     0.682     0.769        22
         5.0      0.840     0.955     0.894        22
         6.0      0.810     0.944     0.872        18
         7.0      0.895     0.944     0.919        18
         8.0      0.429     0.692     0.529        13
         9.0      0.765     0.722     0.743        18

    accuracy                          0.820       200
   macro avg      0.820     0.819     0.813       200
weighted avg      0.840     0.820     0.824       200
```

The LSTM in general seems to have low precision with label 8, which is the reggae genre. In particular, the reggae genre seems to be often confused with the hip hop genre and might be the reason for lower test results. Nevertheless, this might be worth investigating in the future.

### Next Steps:

The most glaring issue is overfitting. Next steps include exploring more ways to mitigate this issue such as changes in architecture or exploring new features. For example, [2] made use of an additional 1D global average pooling layer between the layer normalization and linear layer. Furthermore, [2] used much smaller slices of music (as opposed to a large 3 seconds per slice) as well as the first and second derivatives of some of the features (particularly the MFCCs). Perhaps these methods would answer why the model is overfitting, and be the key to reducing it.

# GRU

### Quantitative Metrics:

We utilize two quantitative metrics to judge the performance of our Gated Recurrent Unit: the (1) Cross Entropy Loss, and (2) the F-measure. We utilize the Cross Entropy Loss to provide gradients to guide training our classifier. The Cross Entropy Loss can be seen as the negated expectation of the predicted posterior of the true genre when using the probability of the true target (i.e., 1). Minimizing this loss leads to a classifier that predicts the correct target distribution for samples within respective genres. We display progress using the F-measure, a harmonic mean of the precision and recall for one target genre averaged over genres. This allows us to ensure genre overrepresentation does not impact class-specific precision.

### Analysis of Algorithm/Model:

We achieved accuracies in line with previous research (90.11%). This is shown in the figure on the next page. Further, the ending F-measure was 0.94, precision was 0.95, and recall was 0.93. We achieved these results after making 10,000,000 updates to a 10 x 28 x 14 n-dimensional numpy array of weights.  Each sample is a one dimensional vector of length 661794. Our model processes input frames in chunks of 7 (for divisibility). Our hidden state is also a vector of length 7. We train our model by randomly selecting 100 tracks from all 100 samples found in the dataset.

With respect to the trained model’s behavior, we found the forget gate was usually neglected throughout the forward pass. This might be due to the distinct ‘sounds’ which define the genre a track belongs to. In other words, the model chooses to cumulatively ingest the track (by turning off the forget gate and outputting a consistently small value from the input gate which gates the current hidden and input states (O(1 / sample length)). 

Interestingly, the output gate starts with a high value, implying the hidden state incorporates more of the current cell memory, then tails off. This further supports the hypothesis that song genres can be predicted with segments smaller than three seconds. We saw this tail off occur at different time points for different genres. This implies some genres are more likely to initially confuse the classifier (ex. Hip Hop and Reggae). 

### Next Steps:

For the Gated Recurrent Unit, we foresee the following next steps:
More Genres. Currently, we only represent ten of the 6000 genres identified on Spotify. To be applicable to the average user, we need to represent diverse interests that may not be captured in the range of classical to rock. To get to 50 genres, we would need to expand the number of songs in our dataset fivefold.
Optimizing Implementation. To train our model for 10,000,000 steps, we needed to wait for almost two hours with a standard-tier Google Cloud CPU. We think we can reduce this time to almost one quarter by using GPUs and parallelizing our training.
Shorter samples. We want to conduct an ablation study to determine the minimum length of a sample required to correctly identify its genre. We project we can use significantly less than three seconds to achieve a similar accuracy of 90.11%.

### Visualizations:
Figure: Display (F-Score) and Training (Cross Entropy) Loss for GRU.

![loss_GRU](https://github.com/darrylj1999/darrylj1999.github.io/blob/main/Figure_1.png)

[1] M. S. Rao, O. Pavan Kalyan, N. N. Kumar, M. Tasleem Tabassum and B. Srihari, "Automatic Music Genre Classification Based on Linguistic Frequencies Using Machine Learning," 2021 International Conference on Recent Advances in Mathematics and Informatics (ICRAMI), Tebessa, Algeria, 2021, pp. 1-5, doi: 10.1109/ICRAMI52622.2021.9585937.

[2] N. N. Wijaya and A. R. Muslikh, “Music-Genre Classification using Bidirectional Long Short-Term Memory and Mel-Frequency Cepstral Coefficients,” in Journal of Computing Theories and Applications 2, vol. 1, pp. 13-26, 2024

[3] Sunil Kumar Prabhakar and Seong-Whan Lee. "Holistic approaches to music genre classification using efficient transfer and deep learning techniques." Expert Systems with Applications, pp. 211, 2023

[4] Eric Odle, Pei-Chun Lin, and Amin Farjudian. "Comparing Recurrent Neural Network Types in a Music Genre Classification Task: Gated Recurrent Unit Superiority Using the GTZAN Dataset."

[5] Leartpantulak, Krittika and Yuttana Kitjaidure. “Music Genre Classification of audio signals Using Particle Swarm Optimization and Stacking Ensemble.” 2019 7th International Electrical Engineering Congress (iEECON) (2019): 1-4.

[6] S. S. Ghosal and I. Sarkar, “Novel approach to music genre classification Authorized licensed use limited to: Georgia Institute of Technology. Downloaded on February 23,2024 at 22:55:57 UTC from IEEE Xplore. Restrictions apply. using clustering augmented learning method (CALM),” in AAAI MAKE, ser. CEUR Workshop Proceedings, vol. 2600, 2020.

[7] L. Yang, J. Hu and Z. Zhang, "Audio Scene Classification Based on Gated Recurrent Unit," 2019 IEEE International Conference on Signal, Information and Data Processing (ICSIDP), Chongqing, China, 2019, pp. 1-5, doi: 10.1109/ICSIDP47821.2019.9173051.

[8] [1] Y. Xu, Q. Kong, Q. Huang, W. Wang, and M. D. Plumbley, “Convolutional gated recurrent neural network incorporating spatial features for audio tagging,” arXiv.org, https://arxiv.org/abs/1702.07787 (accessed Feb. 21, 2024). 

Gantt Chart: [https://docs.google.com/spreadsheets/d/14nsBST_ze4GHueKOfuRy0kKMCA_hBqIP/edit?usp=sharing&ouid=105417584955844239358&rtpof=true&sd=true](https://docs.google.com/spreadsheets/d/14nsBST_ze4GHueKOfuRy0kKMCA_hBqIP/edit?usp=sharing&ouid=105417584955844239358&rtpof=true&sd=true)

Contribution Table
| Name | Midterm Contributions |
| --- | --- |
| Josh | Random Forest |
| Nathan | Preprocessing done for SVM |
| Elias | Set up boilerplate code for training sequential models (LSTM/GRU). Fully implemented BiLSTM. |
| Darryl | GRU |
