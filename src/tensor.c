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
static uint32_t array_prod(uint32_t*, uint32_t);
static void u32reverse(uint32_t*, uint32_t);

static float* slice(float*, uint32_t, uint32_t);
static float* f32copy(float* src, uint32_t n);
static uint32_t* u32copy(uint32_t* src, uint32_t n);

static uint32_t* build_indexer(uint32_t i, uint32_t n_dims, uint32_t* shape);
static float random_uniform(float min, float max);

static void check_n_dims(const tensor_t* t, uint32_t n_dims);

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

tensor_t* tensor_zeros(uint32_t* shape, uint32_t n_dims)
{
    int nels = array_prod(shape, n_dims);
    float* values = (float*)malloc(sizeof(float) * nels);
    for (int i = 0; i < nels; i++)
        values[i] = 0;

    return tensor_new(values, shape, n_dims);
}

tensor_t* tensor_ones(uint32_t* shape, uint32_t n_dims)
{
    tensor_t* t = tensor_zeros(shape, n_dims);
    return tensor_add_scalar(t, 1);
}

tensor_t* tensor_uniform(float min, float max, uint32_t* shape, uint32_t n_dims)
{
    int nels = array_prod(shape, n_dims);
    float* values = (float*)malloc(sizeof(float) * nels);
    for (int i = 0; i < nels; i++)
        values[i] = random_uniform(min, max);

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
    {
        printf("[ERROR] More indices than tensor dimensions\n");
        exit(1);
    }

    result = tensor_copy(t);
    for (int i = 0; i < n_indices; i++)
    {
        if (index[i] >= t->shape[i])
        {
            printf("[ERROR] Invalid index %d for axis %d\n", index[i], i);
            exit(1);
        }

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

tensor_t* tensor_T(const tensor_t* t)
{
    tensor_t* result;
    uint32_t* new_shape;

    check_n_dims(t, 2);
    new_shape = u32copy(t->shape, 2);
    u32reverse(new_shape, 2);
    result = tensor_reshape(t, new_shape, 2);

    for (int i = 0; i < t->shape[0]; i++)
        for (int j = 0; j < t->shape[1]; j++)
            result->values[j * result->shape[1] + i] = t->values[i * t->shape[1] + j];

    return result;
}

tensor_t* tensor_reshape(const tensor_t* t, uint32_t* shape, uint32_t n_dims)
{
    tensor_t* result;

    if (tensor_numel(t) != array_prod(shape, n_dims)) 
    {
        printf("[ERROR] Cannot reshape a tensor of %d elements into  ", tensor_numel(t));
        PRINT_ARRAY(shape, n_dims, "%d", "(", ")", ", ");
        printf(" shape.\n");
        exit(1);
    }
    result = tensor_copy(t);
    result->n_dims = n_dims;
    result->shape = shape;
    return result;
}

void tensor_broadcast(tensor_t** t1, tensor_t** t2)
{
    tensor_t* gc;
    uint32_t* t1_shape, *t2_shape;
    int* broadcasters; 
    uint32_t max_dim = (*t1)->n_dims > (*t2)->n_dims ? (*t1)->n_dims : (*t2)->n_dims;

    /* Align shapes */
    t1_shape = (uint32_t*)malloc(sizeof(uint32_t) * max_dim);
    t2_shape = (uint32_t*)malloc(sizeof(uint32_t) * max_dim);
    broadcasters = (int*)malloc(sizeof(int) * max_dim);

    for (int i = 0; i < max_dim; i++)
    {
        t1_shape[i] = 1;
        t2_shape[i] = 1;
    }

    for (int i = 0; i < (*t1)->n_dims; i++)
        t1_shape[max_dim - i - 1] = (*t1)->shape[(*t1)->n_dims - i - 1];

    for (int i = 0; i < (*t2)->n_dims; i++)
        t2_shape[max_dim - i - 1] = (*t2)->shape[(*t2)->n_dims - i - 1];

    for (int i = 0; i < max_dim; i++)
    {
        if (t1_shape[i] != t2_shape[i] && t1_shape[i] != 1 && t2_shape[i] != 1)
        {
            printf("[ERROR] Cannot broadcast shapes ");
            PRINT_ARRAY((*t1)->shape, (*t1)->n_dims, "%d", "(", ")", ", ");
            printf(" and ");
            PRINT_ARRAY((*t2)->shape, (*t2)->n_dims, "%d", "(", ")", ", ");
            printf("\n");

            free(t1_shape);
            free(t2_shape);
            free(broadcasters);
            exit(1);
        }
        else
        {
            broadcasters[i] = t1_shape[i] - t2_shape[i];
        }
    }

    gc = *t1;
    *t1 = tensor_reshape(*t1, t1_shape, max_dim);
    tensor_clean(gc);

    gc = *t2;
    *t2 = tensor_reshape(*t2, t2_shape, max_dim);
    tensor_clean(gc);

    for (int i = 0; i < max_dim; i++)
    {
        if (broadcasters[i] > 0)
        {
            gc = *t2;
            *t2 = tensor_repeat(*t2, broadcasters[i] + 1, i);
            tensor_clean(gc);
        }
        else if (broadcasters[i] < 0)
        {
            gc = *t1;
            *t1 = tensor_repeat(*t1, -1 * broadcasters[i] + 1, i);
            tensor_clean(gc);
        }
    }
}

tensor_t* tensor_argmax(const tensor_t* t, uint32_t axis)
{
    int pitch;
    int to_reduce;
    int offset = 0, factor;
    float tmp_max, tmp_max_idx;

    uint32_t* new_shape;
    tensor_t* res;

    if (axis > t->n_dims - 1)
    {
        printf("[ERROR] Axis %d is larger than the available n_dims range [0, %d)\n", axis, t->n_dims);
        exit(1);
    }

    pitch = array_prod(&t->shape[axis + 1], t->n_dims - axis - 1);
    to_reduce = t->shape[axis];
    new_shape = (uint32_t*)malloc(sizeof(uint32_t) * t->n_dims - 1);
    for (int i = 0; i < t->n_dims - 1; i++)
    {
        if (i == axis)
            offset = 1;
        new_shape[i] = t->shape[offset + i];
    }

    res = tensor_zeros(new_shape, t->n_dims - 1);
    for (int i = 0; i < tensor_numel(res); i++)
    {
        factor = (int)(i / pitch);
        tmp_max = -99999;
        for (int j = 0; j < to_reduce; j++)
        {
            if (t->values[j * pitch + factor * to_reduce * pitch + (i % pitch)] > tmp_max)
            {
                tmp_max = t->values[j * pitch + factor * to_reduce * pitch + (i % pitch)];
                tmp_max_idx = j;
            }
        }
        res->values[i] = tmp_max_idx;
    }
    return res;
}


tensor_t* tensor_unsqueeze(const tensor_t* t, uint32_t axis)
{
    uint32_t* new_shape;
    int i = 0;
    int offset = 0;

    if (axis > t->n_dims)
    {
        printf("[ERROR] Axis %d is larger than the available n_dims range [0, %d]\n", axis, t->n_dims);
        exit(1);
    }

    new_shape = (uint32_t*)malloc(sizeof(uint32_t) * t->n_dims + 1);
    while (i < t->n_dims + 1)
    {
        if (i == axis)
        {
            new_shape[i] = 1;
            offset = 1;
        }
        new_shape[i + offset] = t->shape[i];
        i++;
    }
    return tensor_reshape(t, new_shape, t->n_dims + 1);
}

tensor_t* tensor_repeat(const tensor_t* t, uint32_t repeats, uint32_t axis)
{
    tensor_t* res;
    int to_repeat, group;

    if (axis > t->n_dims - 1)
    {
        printf("[ERROR] Axis %d is larger than the available n_dims range [0, %d]\n", axis, t->n_dims - 1);
        exit(1);
    }

    res = tensor_copy(t);
    res->shape[axis] *= repeats;
    res->values = (float*)realloc(res->values, sizeof(float) * (int)tensor_numel(res));
    to_repeat = array_prod(&t->shape[axis + 1], t->n_dims - axis - 1);

    for (int i = 0; i < tensor_numel(t); i++)
    {
        group = (int)(i / to_repeat);
        for (int j = group * repeats; j < group* repeats + repeats; j++)
            res->values[j * to_repeat + (i % to_repeat)] = t->values[i];
    }
    return res;
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
        printf("%.4f)\n", t->values[0]);
        return;
    }

    for (int i = 0; i < (int)t->n_dims - 2; i++)
    {
        strcat(trailing_brackets, "]");
        strcat(leading_brackets, "[");
        strcat(leading_space, " ");
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
            printf("%s\n\n%s%s", 
                    trailing_brackets, leading_space, leading_brackets);
        else if (i > 0)
            printf("\n%s ", leading_space);

        PRINT_ARRAY(slice->values, slice_len, "%6.4f", "[", "]", " ");

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


tensor_t* tensor_reduce_sum(const tensor_t* t, uint32_t axis)
{
    tensor_t* result;
    uint32_t* reduced_shape;
    int to_reduce, factor, pitch, offset = 0;
    float tmp;

    if (axis > t->n_dims - 1)
    {
        printf("[ERROR] Axis %d is larger than the available n_dims range [0, %d)\n", axis, t->n_dims);
        exit(1);
    }

    pitch = array_prod(&t->shape[axis + 1], t->n_dims - axis - 1);
    to_reduce = t->shape[axis];
    reduced_shape = (uint32_t*)malloc(sizeof(uint32_t) * t->n_dims - 1);
    for (int i = 0; i < t->n_dims - 1; i++)
    {
        if (i == axis)
            offset = 1;
        reduced_shape[i] = t->shape[offset + i];
    }

    result = tensor_zeros(reduced_shape, t->n_dims - 1);

    for (int i = 0; i < tensor_numel(result); i++)
    {
        factor = (int)(i / pitch);
        tmp = 0;
        for (int j = 0; j < to_reduce; j++)
           tmp += t->values[j * pitch + factor * to_reduce * pitch + (i % pitch)];

        result->values[i] = tmp;
    }


    return result;
}

tensor_t* tensor_gte(const tensor_t* t1, const tensor_t* t2)
{
    tensor_t *o1, *o2;
    o1 = tensor_copy(t1);
    o2 = tensor_copy(t2);

    tensor_broadcast(&o1, &o2);

    tensor_t* res = tensor_copy(o1);
    uint32_t nels = tensor_numel(o1);

    for (int i = 0; i < nels; i++)
        res->values[i] = (float)(o1->values[i] >= o2->values[i]);

    tensor_clean(o1);
    tensor_clean(o2);

    return res;
}

tensor_t* tensor_eq(const tensor_t* t1, const tensor_t* t2)
{
    tensor_t *o1, *o2;
    o1 = tensor_copy(t1);
    o2 = tensor_copy(t2);

    tensor_broadcast(&o1, &o2);

    tensor_t* res = tensor_copy(o1);
    uint32_t nels = tensor_numel(o1);

    for (int i = 0; i < nels; i++)
        res->values[i] = (float)(o1->values[i] == o2->values[i]);

    tensor_clean(o1);
    tensor_clean(o2);

    return res;
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
    tensor_t *o1, *o2;
    o1 = tensor_copy(t1);
    o2 = tensor_copy(t2);

    tensor_broadcast(&o1, &o2);

    tensor_t* res = tensor_copy(o1);
    uint32_t nels = tensor_numel(o1);

    for (int i = 0; i < nels; i++)
        res->values[i] = o1->values[i] + o2->values[i];

    tensor_clean(o1);
    tensor_clean(o2);

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
    tensor_t *o1, *o2;
    o1 = tensor_copy(t1);
    o2 = tensor_copy(t2);

    tensor_broadcast(&o1, &o2);

    tensor_t* res = tensor_copy(o1);
    uint32_t nels = tensor_numel(o1);

    for (int i = 0; i < nels; i++)
        res->values[i] = o1->values[i] * o2->values[i];

    tensor_clean(o1);
    tensor_clean(o2);

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
    tensor_t *o1, *o2;
    o1 = tensor_copy(t1);
    o2 = tensor_copy(t2);

    tensor_broadcast(&o1, &o2);

    tensor_t* res = tensor_copy(o1);
    uint32_t nels = tensor_numel(o1);

    for (int i = 0; i < nels; i++)
        res->values[i] = o1->values[i] / o2->values[i];

    tensor_clean(o1);
    tensor_clean(o2);

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

tensor_t* tensor_mm(const tensor_t* t1, const tensor_t* t2)
{
    if (t1->n_dims != 2 || t2->n_dims != 2)
    {
        printf("[ERROR] tensor_mm only supports tensors of 2 dims.\n");
        exit(1);
    }

    if (t1->shape[t1->n_dims - 1] != t2->shape[0])
    {
        printf("[ERROR] No compatible shapes for matrix multiplication.");
        PRINT_ARRAY(t1->shape, t1->n_dims, "%d", "(", ")", ", ");
        printf(" and ");
        PRINT_ARRAY(t2->shape, t2->n_dims, "%d", "(", ")", ", ");
        printf("\n");
    }

    float tmp;
    uint32_t* shape = (uint32_t*)malloc(sizeof(uint32_t) * 2);
    tensor_t* result;

    shape[0] = t1->shape[0];
    shape[1] = t2->shape[1];
    result = tensor_zeros(shape, 2);

    for (int row = 0; row < t1->shape[0]; row++)
    {
        for (int col = 0; col < t2->shape[1]; col++)
        {
            tmp = 0;
            for (int k = 0; k < t1->shape[1]; k++)
                tmp += t1->values[row * t1->shape[1] + k] * t2->values[k * t2->shape[1] + col];
            result->values[row * result->shape[1] + col] = tmp;
        }
    }
    return result;
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

void u32reverse(uint32_t* arr, uint32_t size)
{
    int tmp;
    int start = 0;
    int end = size - 1;

    while (start < end)
    {
        tmp = arr[start]; 
        arr[start] = arr[end];
        arr[end] = tmp;
        start++;
        end--;
    }
}

void check_n_dims(const tensor_t* t, uint32_t n_dims)
{
    if (t->n_dims != n_dims)
    {
        printf("[ERROR] Expected tensor of %d dims, but got one of %d\n", n_dims, t->n_dims);
        exit(1);
    }
}

float random_uniform(float min, float max)
{
     return min + (float) (rand() / (double) (RAND_MAX) * (max - min));
}

