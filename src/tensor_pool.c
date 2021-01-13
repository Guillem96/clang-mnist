#include "tensor_pool.h"
#include <stdlib.h>

tensor_pool_t* tensor_pool_init()
{
    tensor_pool_t* pool = (tensor_pool_t*)malloc(sizeof(tensor_pool_t));
    pool->ts = (tensor_t**)malloc(sizeof(tensor_t*) * 32);
    pool->capacity = 32;
    pool->len = 0;
    return pool;
}

void tensor_pool_add(tensor_pool_t* pool, const tensor_t* t)
{
    if (pool->len == pool->capacity)
    {
        pool->ts = (tensor_t**)realloc(pool->ts, sizeof(tensor_t*) * pool->capacity * 2);
        pool->capacity = pool->capacity * 2;
    }
    pool->ts[pool->len] = (tensor_t*)t;
    pool->len += 1;
}

void tensor_pool_empty(tensor_pool_t* pool)
{
    for (int i = 0; i < pool->len; i++)
        tensor_clean(pool->ts[i]);
    pool->len = 0;
}

void tensor_pool_clean(tensor_pool_t* pool)
{
    tensor_pool_empty(pool);
    free(pool->ts);
    free(pool);
}
