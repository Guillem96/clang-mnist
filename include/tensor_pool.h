#ifndef _TENSOR_POOL_H_
#define _TENSOR_POOL_H_

#include "tensor.h"

typedef struct 
{
    tensor_t** ts;
    uint32_t capacity;
    uint32_t len;
} tensor_pool_t;


tensor_pool_t* tensor_pool_init();
void tensor_pool_add(tensor_pool_t* pool, const tensor_t* t);
void tensor_pool_empty(tensor_pool_t* pool);
void tensor_pool_clean(tensor_pool_t* pool);

#endif
