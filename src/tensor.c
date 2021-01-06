#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"


#define PRINT_ARRAY(a, l, f, lead, trail, sep) \
    printf(lead);                \
    for (int kk = 0; kk < l; kk++) \
        if (kk < l - 1)          \
            printf(f sep, a[kk]);   \
        else                    \
            printf(f, a[kk]);    \
    printf(trail);

/* Utility functions */
static uint8_t check_equal_shape(const tensor_t* t1, const tensor_t* t2);
static uint32_t array_prod(uint32_t*, uint32_t);
static float* slice(float*, uint32_t, uint32_t);
static float* f32copy(float* src, uint32_t n);
static uint32_t* u32copy(uint32_t* src, uint32_t n);
static uint32_t* build_indexer(uint32_t i, uint32_t n_dims, uint32_t* shape);

tensor_t* tensor_new(float* values, uint32_t* shape, uint32_t n_dims) 
{
    tensor_t* t = (tensor_t*)malloc(sizeof(tensor_t));
    if (n_dims == 0)
        t->shape = NULL;
    else
        t->shape = shape;
    t->values = values;
    t->n_dims = n_dims;
    return t;
}

tensor_t* tensor_arange(
        float start, float end, float step, 
        uint32_t* shape, uint32_t n_dims)
{
    uint32_t nels = (uint32_t)((end - start) / step);
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
    // if (t->n_dims > 0)
        // free(t->shape);
    free(t);
}

tensor_t* tensor_index(const tensor_t* t, uint32_t* index, uint32_t n_indices)
{
    tensor_t* result;
    tensor_t* gc;

    uint32_t next_nels = 0;
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

uint32_t tensor_numel(const tensor_t* t)
{
    if (t->n_dims == 0)
        return 1;

    return array_prod(t->shape, t->n_dims);
}

void tensor_print(const tensor_t* t)
{
    uint32_t length = tensor_numel(t);
    char leading_space[16] = "       ";
    char leading_brackets[16] = "";
    char trailing_brackets[16] = "";

    uint32_t* indexer;
    uint32_t* prev_indexer = NULL;

    tensor_t* slice;
    uint32_t slice_len;

    uint8_t new_group = 0;
    uint8_t is_tensor = t->n_dims > 2;

    printf("Tensor(");

    if (t->n_dims == 0)
    {
        printf("%.2f)\n", t->values[0]);
        return;
    }

    for (int i = 0; i < (int)t->n_dims - 2; i++)
    {
        strcat(trailing_brackets, "]");
        strcat(leading_brackets, "[");
    }

    if (t->n_dims > 1)
        printf("[%s", leading_brackets);

    for (int i = 0; i < length; i += t->shape[t->n_dims  - 1])
    {
        indexer = build_indexer(i, t->n_dims, t->shape);
        slice = tensor_index(t, indexer, t->n_dims - 1);
        slice_len = tensor_numel(slice);
        new_group = prev_indexer != NULL && i > 0 && is_tensor &&
                        (prev_indexer[0] != indexer[0]);

        if (new_group)
            printf("%s\n\n%s %s", 
                    trailing_brackets, leading_space, leading_brackets);
        else if (i > 0)
            printf("\n%s ", leading_space);

        PRINT_ARRAY(slice->values, slice_len, "%6.2f", "[", "]", " ");

        free(prev_indexer);
        prev_indexer = indexer;
        new_group = 0;
    }
    free(prev_indexer);

    if (t->n_dims > 1)
        printf("%s]", trailing_brackets);

    printf(")\n");
}

void tensor_specs(const tensor_t* t)
{
    printf("Tensor(shape=");
    PRINT_ARRAY(t->shape, t->n_dims, "%d", "(", ")", ", ");
    printf(", ndims=%d)\n", t->n_dims);
}


tensor_t* tensor_neg(const tensor_t* t)
{
    tensor_t* res = tensor_copy(t);
    uint32_t nels = tensor_numel(t);
    for (int i = 0; i < nels; i++)
        res->values[i] = -1 * t->values[i];

    return res;
}

tensor_t* tensor_add(const tensor_t* t1, const tensor_t* t2)
{
    if(!check_equal_shape(t1, t2))
        return NULL;

    tensor_t* res = tensor_copy(t1);
    uint32_t nels = tensor_numel(t1);

    for (int i = 0; i < nels; i++)
        res->values[i] = t1->values[i] + t2->values[i];

    return res;
}

tensor_t* tensor_add_scalar(const tensor_t* t, float scalar)
{
    tensor_t* res = tensor_copy(t);
    uint32_t nels = tensor_numel(t);

    for (int i = 0; i < nels; i++)
        res->values[i] = t->values[i] + scalar;

    return res;
}


tensor_t* tensor_sub(const tensor_t* t1, const tensor_t* t2)
{
    tensor_t* neg_t2 = tensor_neg(t2);
    tensor_t* res = tensor_add(t1, neg_t2);
    tensor_clean(neg_t2);
    return res;
}


tensor_t* tensor_sub_scalar(const tensor_t* t, float scalar)
{
    scalar = -1 * scalar;
    return tensor_add_scalar(t, scalar);
}

tensor_t* tensor_mul(const tensor_t* t1, const tensor_t* t2)
{
    if(!check_equal_shape(t1, t2))
        return NULL;

    tensor_t* res = tensor_copy(t1);
    uint32_t nels = tensor_numel(t1);

    for (int i = 0; i < nels; i++)
        res->values[i] = t1->values[i] * t2->values[i];

    return res;
}

tensor_t* tensor_mul_scalar(const tensor_t* t, float scalar)
{
    tensor_t* res = tensor_copy(t);
    uint32_t nels = tensor_numel(t);

    for (int i = 0; i < nels; i++)
        res->values[i] = t->values[i] * scalar;

    return res;
}

tensor_t* tensor_div(const tensor_t* t1, const tensor_t* t2)
{
    if(!check_equal_shape(t1, t2))
        return NULL;

    tensor_t* res = tensor_copy(t1);
    uint32_t nels = tensor_numel(t1);

    for (int i = 0; i < nels; i++)
        res->values[i] = t1->values[i] * t2->values[i];

    return res;
}

tensor_t* tensor_div_scalar(const tensor_t* t, float scalar)
{
    tensor_t* res = tensor_copy(t);
    uint32_t nels = tensor_numel(t);

    for (int i = 0; i < nels; i++)
        res->values[i] = t->values[i] / scalar;

    return res;
}


float* slice(float* old_values, uint32_t start, uint32_t size)
{
    float* new_values = (float*)malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++)
        new_values[i] = old_values[start + i];

    return new_values;
}

float* f32copy(float* src, uint32_t n)
{
    float* new_values = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
        new_values[i] = src[i];

    return new_values;
}

uint32_t* u32copy(uint32_t* src, uint32_t n)
{
    uint32_t* new_values = (uint32_t*)malloc(sizeof(uint32_t) * n);
    for (int i = 0; i < n; i++)
        new_values[i] = src[i];

    return new_values;
}

uint32_t* build_indexer(uint32_t i, uint32_t n_dims, uint32_t* shape)
{
    uint32_t* indexer = (uint32_t*)malloc(
            sizeof(uint32_t) * (n_dims - 1));

    for (int j = 0; j < n_dims - 1; j++) 
    {
        indexer[j] = (uint32_t)(i / array_prod(&shape[j + 1], n_dims - j - 1));
        i -= indexer[j] * array_prod(&shape[j + 1], n_dims - j - 1);
    }

    return indexer;
}

uint32_t array_prod(uint32_t* array, uint32_t n_elems)
{
    int res = 1;
    for (int i = 0; i < n_elems; i++)
        res *= array[i];

    return res;
}


uint8_t check_equal_shape(const tensor_t* t1, const tensor_t* t2)
{
    if (t1->n_dims != t2->n_dims)
        return 0;

    for (int i = 0; i < t1->n_dims; i++)
        if (t1->shape[i] != t2->shape[i])
            return 0;

    return 1;
}

