#include <stdio.h>

#include "tensor.h"
#include "mnist.h"
#include "plot.h"

#define TRAIN_IMAGES "data/train-images.idx3-ubyte"
#define TRAIN_LABELS "data/train-labels.idx1-ubyte"
#define TEST_IMAGES "data/t10k-images.idx3-ubyte"
#define TEST_LABELS "data/t10k-labels.idx1-ubyte"

int main() {
    unsigned int shape[] = {3, 3};
    unsigned int indexer[] = {1};
    char title[256];

    tensor_t* m = tensor_arange(0, 9, 1, shape, 2);

    tensor_print(m);
    tensor_specs(m);

    tensor_t* indexed = tensor_index(m, indexer, 1);
    tensor_print(indexed);
    tensor_specs(indexed);

    tensor_t* added = tensor_add(m, m);
    tensor_print(added);

    tensor_t* added_s = tensor_add_scalar(m, 1.5);
    tensor_print(added_s);

    tensor_t* sub = tensor_sub(m, m);
    tensor_print(sub);

    tensor_t* mul = tensor_mul(m, m);
    tensor_print(mul);

    tensor_clean(indexed);
    tensor_clean(added);
    tensor_clean(added_s);
    tensor_clean(sub);
    tensor_clean(mul);
    tensor_clean(m);

    mnist_t* ds = mnist_read(TRAIN_IMAGES, TRAIN_LABELS);

    mnist_example_t* sample = mnist_sample(ds, 0);
    tensor_specs(sample->image);
    sprintf(title, "This is a %d", (int)sample->label->values[0]);
    imshow(sample->image, title);

    sample = mnist_sample(ds, 0);
    sprintf(title, "This is a %d", (int)sample->label->values[0]);
    imshow(sample->image, title);

    mnist_clean(ds);
    return 0;
}
