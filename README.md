# Music Genre Classification

Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

test

## Dataset Description:
Our dataset comprises 3-second excerpts from songs spanning 10 distinct genres. Each data point is characterized by 60 features, encompassing key audio metrics such as the root mean square value, spectral centroid value, and zero-crossing rate. These are all metrics used within the music industry to quantify sound. Each of the 10 genres is represented by 100 songs, and each song is further divided into 10 separate 3-second excerpts. This division ensures a comprehensive and varied representation of each genre, contributing to the diversity of our dataset.

## Literature Review
Previous papers have shown various approaches to classifying and predicting genre of songs from musical features. For example, Leartpantulak et al. have shown a stacking ensemble method which involves two classifiers, a base (K-Nearest Neighbors (k-NN), Decision Tree (DT), Random Forest, Support Vector Machines (SVM); and Naïve Bayes) which classifies and is then sent to a meta classifier [5]. Other groups, such as Ghosal et al. employed convolutional neural networks to extract features and utilized LSTM autoencoders to capture temporal dynamics when analyzing time series data [6]. We found that Gated Recurrent Units outperformed other models at the generic audio and audio scene classification tasks [7]. In fact, more layers (ex. Convolutions) can be incorporated to extend superior performance with multimodal data [8]. 

## Problem Definition
The problem at hand revolves around predicting the music genre for different songs given only audio characteristics. It can be very labor-intensive to manually match songs to a certain genre, and thousands of new songs are being released everyday. Our goal is to automate this process by developing machine learning models capable of classifying music into genres using quantitative features extracted from audio signals.

There are a couple reasons that we are motivated to address this problem. First, the sheer volume and diversity of music available today make manual genre classification a daunting task. An automated system would expedite the categorization process, aiding streaming platforms in organizing music more efficiently. 

Second, a genre prediction model contributes to enhancing user experiences in music recommendation systems. By understanding the underlying features that define different genres, personalized playlists and recommendations can be tailored to individual preferences. 

## Methods
Data Preprocessing:
Normalization (Min-max scaling or normal scaling)
Label Encoding of genres
PCA (unsupervised)
For feature reduction
Mel-Frequency Cepstral Coefficients (MFCC) extraction [2]
Simulates human hearing when listening to music [2]

## Models:
Random Forest [2], [3]
Shown to have good accuracy (83%) by previous researchers [3]
SVM [2], [3]
Shown to have decent accuracy (78%) by previous researchers [3]
BiLSTM [2]
Current state of the art with this dataset, boasting the highest accuracies and F1 score (both about 94%) [2]
GRU [4]
Current state of the art with this dataset, with very high accuracy (90%) [4]

## Results and Discussion
Quantitative Metrics:
Accuracy [1]
Frequently used metric in papers regarding this dataset
F1 Score [2]
Frequently used metric in papers regarding this dataset
AUC
Gives an idea of which type of model does best in this domain

## Project Goals:
Achieve an accuracy of at least 60-70% [1] [2]
Achieve F1 Score of at least 60% [2]

## References

1. M. S. Rao, O. Pavan Kalyan, N. N. Kumar, M. Tasleem Tabassum and B. Srihari, "Automatic Music Genre Classification Based on Linguistic Frequencies Using Machine Learning," 2021 International Conference on Recent Advances in Mathematics and Informatics (ICRAMI), Tebessa, Algeria, 2021, pp. 1-5, doi: 10.1109/ICRAMI52622.2021.9585937.

2. N. N. Wijaya and A. R. Muslikh, “Music-Genre Classification using Bidirectional Long Short-Term Memory and Mel-Frequency Cepstral Coefficients,” in Journal of Computing Theories and Applications 2, vol. 1, pp. 13-26, 2024

3. Sunil Kumar Prabhakar and Seong-Whan Lee. "Holistic approaches to music genre classification using efficient transfer and deep learning techniques." Expert Systems with Applications, pp. 211, 2023

4. Eric Odle, Pei-Chun Lin, and Amin Farjudian. "Comparing Recurrent Neural Network Types in a Music Genre Classification Task: Gated Recurrent Unit Superiority Using the GTZAN Dataset."

5. Leartpantulak, Krittika and Yuttana Kitjaidure. “Music Genre Classification of audio signals Using Particle Swarm Optimization and Stacking Ensemble.” 2019 7th International Electrical Engineering Congress (iEECON) (2019): 1-4.

6. S. S. Ghosal and I. Sarkar, “Novel approach to music genre classification Authorized licensed use limited to: Georgia Institute of Technology. Downloaded on February 23,2024 at 22:55:57 UTC from IEEE Xplore. Restrictions apply. using clustering augmented learning method (CALM),” in AAAI MAKE, ser. CEUR Workshop Proceedings, vol. 2600, 2020.

7. L. Yang, J. Hu and Z. Zhang, "Audio Scene Classification Based on Gated Recurrent Unit," 2019 IEEE International Conference on Signal, Information and Data Processing (ICSIDP), Chongqing, China, 2019, pp. 1-5, doi: 10.1109/ICSIDP47821.2019.9173051.

8. Y. Xu, Q. Kong, Q. Huang, W. Wang, and M. D. Plumbley, “Convolutional gated recurrent neural network incorporating spatial features for audio tagging,” arXiv.org, https://arxiv.org/abs/1702.07787 (accessed Feb. 21, 2024). 

## Contribution Table

| Name | Contribution |
|------|--------------|
| Josh | Authored a Dataset Description, procured reference [1], identified quantitative metrics and project goals, summarized the problem into into a problem statement, proposed Random Forest and SVM models to ensure a diversified approach to genre classification |
| Nathan | Wrote part and researched part of literature review, procured references [5] and [6], set up github repository |
| Elias | Found music genre classification dataset, procured references [2] and [3], proposed and justified PCA preprocessing, proposed and justified with research MFCC preprocessing, proposed and justified with research KNN model, proposed and justified with research BiLSTM model, set project goals based on references, confirmed that random forest and SVM have been previously used in this domain’s research, scheduled meetings |
| Darryl | Added two references to Gated Recurrent Units and summarized their contents in Literature Review. Publish Github Page using Markdown to ensure our results are viewable in web format to the teaching team. |

