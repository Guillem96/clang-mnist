#ifndef _MATRIX_H_
#define _MATRIX_H_

typedef struct {
    unsigned int n_dims;
    unsigned int* shape;
    float* values;
} tensor_t;

/* Factory methods */
tensor_t* tensor_new(float* values, unsigned int* shape, unsigned int n_dims);
tensor_t* tensor_arange(float start, float end, float step, unsigned int* shape, unsigned int n_dims);
tensor_t* tensor_copy(const tensor_t* t);
tensor_t* tensor_index(const tensor_t* t, unsigned int* index, unsigned int n_indices);


/* Destructor */
void tensor_clean(tensor_t* t);

unsigned int tensor_numel(const tensor_t* t);

void tensor_print(const tensor_t* t);
void tensor_specs(const tensor_t* t);

#endif
