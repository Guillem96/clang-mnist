#ifndef _MNIST_H_
#define _MNIST_H_

#include <stdint.h>
#include "tensor.h"

typedef struct 
{
    int magic_number;
    int n_items;
    unsigned char* labels;
} mnist_labels_t;

typedef struct 
{
    int magic_number;
    int n_images;
    int rows;
    int cols;
    unsigned char* pixels;
} mnist_images_t;

typedef struct 
{
    mnist_images_t *images;
    mnist_labels_t *labels;
} mnist_t;

typedef struct
{
    tensor_t* image;
    tensor_t* label;
} mnist_example_t;

mnist_t* mnist_read(const char* images_fname, const char* labels_fname);
mnist_example_t* mnist_sample(mnist_t* ds, uint8_t flat);
mnist_example_t* mnist_batch(mnist_t* ds, int n_samples, uint8_t flat);

void mnist_clean(mnist_t* ds);

#endif
