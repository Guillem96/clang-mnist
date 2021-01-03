#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

/* Utility functions */
static unsigned int array_prod(unsigned int*, unsigned int);
static float* slice(float*, unsigned int, unsigned int);
static float* f32copy(float* src, unsigned int n);
static unsigned int* u32copy(unsigned int* src, unsigned int n);
static unsigned int* build_indexer(
        unsigned int i, unsigned int n_dims, 
        unsigned int* shape);

tensor_t* tensor_new(
        float* values, 
        unsigned int* shape, unsigned int n_dims) 
{
    tensor_t* t = (tensor_t*)malloc(sizeof(tensor_t));
    t->shape = shape;
    t->values = values;
    t->n_dims = n_dims;
    return t;
}

tensor_t* tensor_arange(
        float start, float end, float step, 
        unsigned int* shape, unsigned int n_dims)
{
    unsigned int nels = (unsigned int)((end - start) / step);
    float* values = (float*)malloc(sizeof(float) * nels);
    int i = 0;
    for (float v = start; v < end; v += step)
    {
        values[i] = v;
        i++;
    }
    return tensor_new(values, shape, n_dims);
}

tensor_t* tensor_copy(const tensor_t* t) 
{
    return tensor_new(
            f32copy(t->values, tensor_numel(t)), 
            u32copy(t->shape, t->n_dims), t->n_dims);
}

void tensor_clean(tensor_t* t)
{
    free(t->values);
    free(t->shape);
    free(t);
}

tensor_t* tensor_index(const tensor_t* t, unsigned int* index, unsigned int n_indices)
{
    tensor_t* result;
    tensor_t* gc;

    unsigned int next_nels = 0;
    if (n_indices > t->n_dims)
        return NULL;

    result = tensor_copy(t);
    for (int i = 0; i < n_indices; i++)
    {
        if (i < t->n_dims - 1) 
            next_nels = array_prod(&t->shape[i + 1], t->n_dims - i - 1);
        else
            next_nels = 1;
        gc = result;
        result = tensor_new(
                slice(gc->values, 
                      next_nels * index[i], 
                      next_nels), 
                u32copy(&t->shape[i + 1], t->n_dims - i - 1),
                t->n_dims - i - 1);
        tensor_clean(gc);
    }
    return result;
}

unsigned int tensor_numel(const tensor_t* t)
{
    if (t->n_dims == 0)
        return 1;

    return array_prod(t->shape, t->n_dims);
}

void tensor_print(const tensor_t* t)
{
    unsigned int length = tensor_numel(t);
    char leading_space[16] = "       ";
    unsigned int* indexer;
    unsigned int* prev_indexer = NULL;

    tensor_t* slice;
    unsigned int slice_len;

    int new_group = 0;
    int is_tensor = t->n_dims > 2;

    printf("Tensor(");

    if (t->n_dims == 0)
    {
        printf("%.2f)\n", t->values[0]);
        return;
    }
    
    for (int i = 0; i < (int)t->n_dims - 2; i++)
    {
        printf("[");
        strcat(leading_space, " ");
    }

    for (int i = 0; i < length; i += t->shape[t->n_dims  - 1])
    {
        indexer = build_indexer(i, t->n_dims, t->shape);
        slice = tensor_index(t, indexer, t->n_dims - 1);
        slice_len = tensor_numel(slice);
        new_group = prev_indexer != NULL && i > 0 && is_tensor &&
                        (prev_indexer[0] != indexer[0]);

        if (new_group)
            printf("]\n\n%s[", leading_space);
        else if (i > 0)
            printf("\n%s ", leading_space);
        else if (t->n_dims > 1)
            printf("[");

        printf("[");

        for (int j = 0; j < slice_len; j++)
            if (j < slice_len - 1)
                printf("%.2f ", slice->values[j]);
            else
                printf("%.2f", slice->values[j]);

        printf("]");

        free(prev_indexer);
        prev_indexer = indexer;
        new_group = 0;
    }
    free(prev_indexer);

    for (int i = 0; i < t->n_dims - 1; i++)
    {
        printf("]");
    }

    printf(")\n");
}

void tensor_specs(const tensor_t* t)
{
    printf("Tensor(shape=");

    /* Print shape */
    printf("(");
    for (int i = 0; i < t->n_dims; i++)
        if (i < t->n_dims - 1)
            printf("%d, ", t->shape[i]);
        else
            printf("%d", t->shape[i]);
    printf("), ");

    printf("ndims=%d)\n", t->n_dims);
}

static unsigned int array_prod(unsigned int* array, unsigned int n_elems)
{
    int res = 1;
    for (int i = 0; i < n_elems; i++)
        res *= array[i];
    return res;
}

static float* slice(float* old_values, unsigned int start, unsigned int size)
{
    float* new_values = (float*)malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++)
    {
        new_values[i] = old_values[start + i];
    }
    return new_values;
}

static float* f32copy(float* src, unsigned int n)
{
    float* new_values = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
    {
        new_values[i] = src[i];
    }
    return new_values;
}

static unsigned int* u32copy(unsigned int* src, unsigned int n)
{
    unsigned int* new_values = (unsigned int*)malloc(sizeof(unsigned int) * n);
    for (int i = 0; i < n; i++)
    {
        new_values[i] = src[i];
    }
    return new_values;
}

static unsigned int* build_indexer(
        unsigned int i, unsigned int n_dims, 
        unsigned int* shape)
{
    unsigned int* indexer = (unsigned int*)malloc(
            sizeof(unsigned int) * (n_dims - 1));

    for (int j = 0; j < n_dims - 1; j++) 
    {
        indexer[j] = (unsigned int)(i / array_prod(&shape[j + 1], n_dims - j - 1));
        i -= indexer[j] * array_prod(&shape[j + 1], n_dims - j - 1);
    }

    return indexer;
}

