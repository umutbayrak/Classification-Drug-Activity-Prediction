# Classification-Drug-Activity-Prediction
The project for the Big Data Science course

The dataset is not available due to confidentiality.

# Classification: Drug activity prediction

In this part of the project, you will be presented with a challenging classfication problem regarding drug activity prediction. The training data set consists of 1909 compounds tested for their ability to bind to a target site on thrombin, a key receptor in blood clotting. Of these compounds, 42 are active (bind well) and the others are inactive. Each compound is described by a single feature vector comprised of a class value (A for active, I for inactive) and 139,351 binary features, which describe three-dimensional structural properties of the molecule. The dataset consists of two parts: a training set (with known class labels), which you can use to build and optimize your model, and a test set (without class labels), on which you are expected to provide us with your best possible predictions. This dataset poses a number of challenges, which will require you to experiment with a number of different approaches:

 The data set is high-dimensional, and contains a lot more features than samples. This calls for special care to avoid overfitting, e.g. by using feature selection algorithms and (strong) regularization.
 The data set is very imbalanced, so your model should avoid predicting everything as the majority class. To obtain valid models, you should thus use appropriate evaluations measures (e.g. ROC based measures, balanced accuracy,...) and possibly specific sampling approaches to deal with imbalanced data.
 The dataset is difficult, so do not expect too high values of your performance measures. A balanced accuracy of > 60% is already a very good result. You are required to deliver at least the following models:

1. Compare at least three different classification models on the training set, and compare their performance using a validation approach of your choice (e.g. cross- validation, repeated splitting in train/test portions,...). As performance measure choose at least two different measures that are able to work well in the case of imbalanced data.
2. Explore at least three feature selection mechanisms and compare your model (using the same validation approach as in the previous part) to the baseline models without feature selection. You can either choose specific feature selection methods (e.g. filter approaches), or use model-based approaches such as wrapper or embedded feature selection approaches (e.g. Random Forest based importance values, regularization,...).
3. Explore at least two ways to improve your model for imbalanced data classification. Examples include sampling based approaches (e.g. SMOTE), or class weighing approaches.
4. Finally choose your best model configuration and train it on the full training set, in order to produce predictions for the test set provided. Along with your report, you should provide us with your predictions for the test set (in the same order as they occur in the test set file).
