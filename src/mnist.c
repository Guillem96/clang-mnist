#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/* Utility functions  */
static int reverse_int(int i);
static void read_int(FILE* f, int* value);

mnist_t* mnist_read(const char* images_fname, const char* labels_fname)
{
    FILE* images_f;
    FILE* labels_f;
    int kk, amount;

    mnist_images_t* mnist_images = (mnist_images_t*)malloc(sizeof(mnist_images_t));
    mnist_labels_t* mnist_labels = (mnist_labels_t*)malloc(sizeof(mnist_labels_t));
    mnist_t* mnist = (mnist_t*)malloc(sizeof(mnist_t));

    images_f = fopen(images_fname, "rb");
    labels_f = fopen(labels_fname, "rb");

    if (images_f == NULL)
    {
        printf("[ERROR] Reading file %s\n", images_fname);
        return NULL;
    }

    if (labels_f == NULL)
    {
        printf("[ERROR] Reading file %s\n", labels_fname);
        return NULL;
    }

    /* Filling images struct  */
    fread(&kk, 4, 1, images_f);
    read_int(images_f, &(mnist_images->n_images));
    read_int(images_f, &(mnist_images->rows));
    read_int(images_f, &(mnist_images->cols));

    amount = mnist_images->n_images * mnist_images->rows * mnist_images->cols;
    mnist_images->pixels = (unsigned char*)malloc(amount);

    if (fread(mnist_images->pixels, 1, amount, images_f) != amount)
    {
        printf("[ERROR] Reading bytes from images file\n");
        return NULL;
    }

    /* Filling labels struct  */
    fread(&kk, 4, 1, labels_f);
    read_int(labels_f, &(mnist_labels->n_items));
    mnist_labels->labels = (unsigned char*)malloc(mnist_labels->n_items);

    if (fread(mnist_labels->labels, 1, 
                mnist_labels->n_items, labels_f) != mnist_labels->n_items)
    {
        printf("[ERROR] Reading bytes from labels file\n");
        return NULL;
    }

    mnist->images = mnist_images;
    mnist->labels = mnist_labels;

    fclose(images_f);
    fclose(labels_f);
    return mnist;
}

mnist_example_t* mnist_sample(mnist_t* ds, uint8_t flat)
{
    int rand_idx = rand() % ds->images->n_images;

    /* Sample image variables */
    int n_bytes = ds->images->rows * ds->images->cols;
    int base_idx = n_bytes * rand_idx;
    int n_dims = flat ? 1 : 2;
    uint32_t* shape = (uint32_t*)malloc(sizeof(uint32_t) * n_dims);
    float* values = (float*)malloc(sizeof(float) * n_bytes);

    /* Sample label variables */
    float* label_values = (float*)malloc(sizeof(float));
    label_values[0] = (float)ds->labels->labels[rand_idx];

    mnist_example_t* result = (mnist_example_t*)malloc(sizeof(mnist_example_t));

    if (flat) 
    {
        shape[0] = n_bytes;
    } 
    else
    {
        shape[0] = ds->images->rows;
        shape[1] = ds->images->cols;
    }

    for (int i = 0; i < n_bytes; i++)
        values[i] = (float)ds->images->pixels[base_idx + i];

    result->image = tensor_new(values, shape, n_dims);
    result->label = tensor_new(label_values, NULL, 0);
    return result;
}

mnist_example_t* mnist_batch(mnist_t* ds, int n_samples, uint8_t flat)
{
    return NULL;
}

void mnist_clean(mnist_t* ds)
{
   free(ds->images->pixels);
   free(ds->images);
   free(ds->labels->labels);
   free(ds->labels);
   free(ds);
}

void read_int(FILE *f, int* value)
{
    fread(value, sizeof(int), 1, f);
    *value = reverse_int(*value);
}

int reverse_int(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

