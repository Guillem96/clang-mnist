#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdint.h>

typedef struct {
    uint32_t n_dims;
    uint32_t* shape;
    float* values;
} tensor_t;

/* Factory methods */
tensor_t* tensor_new(float* values, uint32_t* shape, uint32_t n_dims);
tensor_t* tensor_arange(float start, float end, float step, uint32_t* shape, uint32_t n_dims);
tensor_t* tensor_copy(const tensor_t* t);
tensor_t* tensor_index(const tensor_t* t, uint32_t* index, uint32_t n_indices);

/* Destructor */
void tensor_clean(tensor_t* t);

uint32_t tensor_numel(const tensor_t* t);
void tensor_reshape(tensor_t* t);

/* Basic math operations */
tensor_t* tensor_neg(const tensor_t* t);

tensor_t* tensor_add(const tensor_t* t1, const tensor_t* t2);
tensor_t* tensor_add_scalar(const tensor_t* t, float scalar);

tensor_t* tensor_sub(const tensor_t* t1, const tensor_t* t2);
tensor_t* tensor_sub_scalar(const tensor_t* t, float scalar);

tensor_t* tensor_mul(const tensor_t* t1, const tensor_t* t2);
tensor_t* tensor_mul_scalar(const tensor_t* ti, float scalar);

tensor_t* tensor_div(const tensor_t* t1, const tensor_t* t2);
tensor_t* tensor_div_scalar(const tensor_t* t, float scalar);

/* Standard output information */
void tensor_print(const tensor_t* t);
void tensor_specs(const tensor_t* t);

#endif
