# MNIST from Scratch 🧠🔢:w

This repository implements an end2end Neural Network (forward and backward) in c.

The goal is to train a NN able to recognize digits.

## Download MNIST

Just navigate inside data directory and run:

$ sh download.sh

## Tensor module

To make my solution more scalable a I created a reusable tensor module. This module
allows the user using the API perform basic operations in numpy fashion.

```c
#include "tensor.h"
#include "tensor_pool.h"

uint32_t shape[] = {3, 1};
uint32_t input_shape = {10, 3};

tensor_pool_t* pool = tensor_pool_init();

tensor_t* X = tensor_arange(0, 30, 1, input_shape, 2);
tensor_t* W = tensor_uniform(-1, 1, shape, 2);
tensor_t* b = tensor_zeros(&shape[1], 1);

tensor_t* linear_regression = tensor_mm(X, W);
tensor_pool_add(pool, linear_regression); // Keep intermediate pointers to later free them
linear_regression = tensor_add(W, b); // Also supporting BROADCASTING!!! 😲

tensor_specs(linear_regression); // Out: Tensor(shape=(10, 1), n_dims=2)
tensor_print(linear_regression);
tensor_pool_clean(pool);

tensor_clean(X); 
// .. clean all tensors from memory
```

## MNSIT & Plot module 📉📊

Plots are cool, so why not implementing a function to plot images.

```c
#include "mnist.h"
#include "tensor.h"
#include "plot.h"

char* images_path = "....";
char* labels_poth = "....";
char title[256];

mnist_t* ds = mnist_read(images_path, labels_path);
mnist_example_t* sample = mnist_sample(ds, 0);

sprintf(title, "This is %d", sample->label);
imshow(sample->image, title);

mnist_clean(ds);
tensor_clean(sample->image);
tensor_clean(sample->label);
```
