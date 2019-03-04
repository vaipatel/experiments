# MDN for Inverse Kinematics Data using Keras MDN

This experiment is an attempt at reproducing the Robot Inverse Kinematics example presented in the incredible [Mixture Density Networks paper (1994) by C. M. Bishop](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.5685&rep=rep1&type=pdf) using the [Keras MDN Layer](https://github.com/cpmpercussion/keras-mdn-layer) (which relies [TensorFlow Probability](https://www.tensorflow.org/probability)).

All embedded images are from the 1994 paper itself.

STATUS: Failing to reproduce the result of improvement in position prediction of the robot arm's end-effector using a Mixture Density Network. But successfully (IMO) reproduced the simple MLP baseline attempted prior to the MDN.

## Report

I am not doing a good job of reproducing the main MDN result from the paper. The MDN seems to learn the distinction between the 3 "regions" - A, B and C. But it's not approximating the region B and the hard edges very well. Also while the $\theta$'s seem to be in the general regions, they are quite displaced from their true positions.

Possible reasons for failure to reproduce:
1. Something wrong in my implementation.
2. Something wrong in my implementation.
3. Maybe Adam is not good here, and BFGS should have been used.
4. Seems like the model is still learning even at the end of training, so maybe just increase to like 1 million epochs (lol).
5. Possibly the Gaussian kernel, but the paper did just fine with it.

## Implementation notes

### Differences from paper

1. I use 3000 points instead of 1000.
2. For training both the preliminary feed-forward NN and the MDN, I use the AdamOptimizer instead of BFGS.
3. For the feed-forward NN, I used 10000 epochs instead of 3000 cycles used in the paper.
4. For the MDN, I use 40 hidden units, 40 output units, 64 mixture components,  instead of 10 hidden units, 8 output units, 2 mixture components used in the paper. Also tried custom losses where I linearly combined the likelihood loss and the squared loss, even annealing the weight (down and up respectively).

### Coding specifics

1. I implement the mixture model using the [Keras MDN Layer](https://github.com/cpmpercussion/keras-mdn-layer) (which relies TensorFlow Probability).
    
    Underneath the hood, the Keras MDN layer leverages [Categorical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical) for the mixing coefficients, [MultivariateNormalDiag](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiag) for constructing multivariate gaussian components from means and diagonal covariance matrices, and [Mixture](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Mixture) for constructing the mixture model using the probabilities and components. This spares us from having to code up things like the Bivariate Gaussian kernel, the Mixture sampling etc. because we can just defer to TFP for these things. On the downside, we probably
 1. lose some efficiency because we are not providing a closed-form version of the kernel ourselves,
 2. lose some control over abating numerical precision problems, and so every once in a while during training/evaluation we will get an error due to a malformed covariance matrix. (But there are tricks in [Alex Brando's Masters Thesis](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation) to help alleviate this challenge, which are employed in Keras MDN Layer.)