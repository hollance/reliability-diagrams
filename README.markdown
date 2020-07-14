# Reliability diagrams

A classifier with a sigmoid or softmax layer outputs a number between 0 and 1 for each class, which we tend to interpret as the **probability** that this class was detected. However, this is only the case if the classifier is **calibrated** properly!

The paper [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) by Guo et al. (2017) claims that modern, deep neural networks are often not calibrated. As a result, interpreting the predicted numbers as probabilities is not correct.

When a model is calibrated, the confidence score should equal the accuracy. For example, if your test set has 100 examples for which the model predicts 0.8, the accuracy over those 100 examples should be 80%. In other words, if 0.8 is a true probability, the model should get 20% of these examples wrong! For all the examples with confidence score 0.9, the accuracy should be 90%, and so on.

One way to find out how well your model is calibrated, is to draw a **reliability diagram**. Here is the reliability diagram for the model **gluon_senet154**, from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), trained on ImageNet:

![](figures/gluon_senet154_ImageNet_pytorch-image-models.png)

These results were computed over the ImageNet validation set of 50,000 images.

The top part of this image is the reliability diagram. The bottom part is a confidence histogram.

## How to interpret these plots?

First, the model's predictions are divided into bins based on the confidence score of the winning class. For each bin we calculate the average confidence and average accuracy.

The **confidence histogram** at the bottom shows how many test examples are in each bin. Here, we used 20 bins. It is clear from the histogram that most predictions of this model had a confidence of > 0.8.

The two vertical lines indicate the overall accuracy and average confidence. The closer these two lines are together, the better the model is calibrated.

The **reliability diagram** at the top shows the average confidence for each bin, as well as the accuracy of the examples in each bin.

Usually the average confidence for a bin lies on or close to the **diagonal**. For example, if there are 20 bins, each bin is 0.05 apart. Then the average confidence for the bin (0.9, 0.95] will typically be around 0.925. (The first bin is an exception: with softmax, the probability of the winning prediction must be at least larger than `1/num_classes`, so this pushes the average confidence up a bit for that bin.)

For each bin we plot the difference between the accuracy and the confidence. Ideally, the accuracy and confidence are equal. In that case, the model is calibrated and we can interpret the confidence score as a probability.

If the model is not calibrated, however, there is a **gap** between accuracy and confidence. These are the red bars in the diagram. The larger the bar, the greater the gap.

The diagonal is the ideal accuracy for each confidence level. If the red bar goes below the diagonal, it means the confidence is larger than the accuracy and the model is too confident in its predictions. If the red bar goes above the diagonal, the accuracy is larger than the confidence, and the model is not confident enough.

The **black lines** in the plot indicate the average accuracy for the examples in that bin:

- If the black line is at the bottom of a red bar, the model is over-confident for the examples in that bin.

- If the black line is on top of a red bar, the model is not confident enough in its predictions.

By calibrating the model, we can bring these two things more in line with one another. Note that, when calibrating, the model's accuracy doesn't change (although this may depend on the calibration method used). It just fixes the confidence scores so that a prediction of 0.8 really means the model is correct 80% of the time.

For the **gluon_senet154** plot above, notice how most of the gaps extend above the diagonal. This means the model is more accurate than it thinks. Only for the bin (0.95, 1.0] is it overestimating its accuracy. For the bins around 0.5, the calibration is just right.

Because not every bin has the same number of examples, some bins affect the calibration of the model more than others. You can see this distribution in the histrogram. To make the importance of the bins even clearer, the red bars are darker for bins with more examples and lighter for bins with fewer examples. It's immediately clear that most predictions for this model have a confidence between 0.85 and 0.9, but that the accuracy for this bin is actually more like ~0.95.

The top diagram also includes the **ECE** or Expected Calibration Error. This is a summary statistic that gives the difference in expectation between confidence and accuracy. In other words, it's an of the gaps across all bins, weighed by the number of examples in each bin. Lower is better.

## Is it bad to be more accurate than confident?

The models in the Guo paper are *too* confident. On examples for which those models predict 90% confidence, the accuracy is something like 80%. That obviously sounds like it's a problem.

But in my own tests, so far, I found that the accuracy is actually larger than the confidence in most bins, meaning the model is underestimating the confidence scores. You can see this in the plot above: the black lines (indicating the accuracy) almost all lie above the diagonal. 

Is that a bad thing? A model being more accurate than it is confident doesn't sound so bad... 

So, which is worse: a model that is too confident, or a model that is not confident enough?

- Over-confidence (conf > acc): this gives more **false positives**. You make observations that are not actually true. This is the same as setting your decision threshold lower, so more comes through.

- Under-confidence (acc > conf): this gives more **false negatives**. This is as if you're using a higher decision threshold, so you lose more observations.

Which is worse depends on the use case, I guess. But if you want to be able to properly interpret the predictions as probabilities -- for example, because you want to feed the output from the neural network into another probabilistic model -- then you don't want the gaps in the reliability diagram to be too large. Remember that calibrating doesn't change the accuracy, it just shifts the confidences around so that 0.9 really means you get 90% correct.

## OK, so my model is not calibrated, how do I fix it?

See the Guo paper for techniques. There have also been follow-up papers with new techniques.

The goal of this repo is only to help visualize whether a model needs calibration or not. Think of it as being a part of a diagnostic toolkit for machine learning models.

## The code

Python code to generate these diagrams is in [reliability_diagrams.py](reliability_diagrams.py). 

The notebook [Plots.ipynb](Plots.ipynb) shows how to use it.

The folder [results](results/) contains CSV files with the predictions of various models. The CSV file has three columns:

```
true_label, pred_label, confidence
```

For a multi-class model, the predicted label and the confidence are for the highest-scoring class.

To generate a reliability diagram for your own model, run it on your test set and output a CSV file in this format.

Currently included are results for models from:

- [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [torchvision](https://pytorch.org/docs/stable/torchvision/models.html)
- [markus93/NN_calibration](https://github.com/markus93/NN_calibration) -- I also used ideas from their source code
- "snacks", the model trained in my book [Machine Learning by Tutorials](https://store.raywenderlich.com/products/machine-learning-by-tutorials)

Interestingly, the models from pytorch-image-models tend to under-estimate the confidence, while similar models from torchvision are overconfident (both are trained on ImageNet).

The [figures](figures/) folder contains some PNG images of these reliability diagrams.

License: MIT
