#include "nn.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

tensor_t* nn_relu(tensor_t* t)
{
    tensor_t* result = tensor_copy(t);
    for (int i = 0; i < tensor_numel(t); i++)
        result->values[i] = t->values[i] > 0 ? t->values[i] : 0;
    return result;
}

tensor_t* nn_softmax(tensor_t* t, uint32_t axis)
{
    tensor_t* exp_tensor = tensor_copy(t);
    tensor_t* activation;
    tensor_t* denominator;
    tensor_t* gc;

    for (int i = 0; i < tensor_numel(t); i++)
        exp_tensor->values[i] = exp(exp_tensor->values[i]);

    denominator = tensor_reduce_sum(exp_tensor, axis);
    gc = denominator;

    denominator = tensor_unsqueeze(denominator, axis);
    activation = tensor_div(exp_tensor, denominator);

    tensor_clean(denominator);
    tensor_clean(gc);
    tensor_clean(exp_tensor);

    return activation;
}

float nn_sparse_ce_loss(const tensor_t* y_true, const tensor_t* y_pred)
{
    float loss = 0;

    if (y_true->n_dims != 1)
    {
        printf("[ERROR] y_true must have a single dim.\n");
        exit(-1);
    }

    if (y_pred->n_dims != 2)
    {
        printf("[ERROR] y_pred must have 2 dimensions.\n");
        exit(-1);
    }

    if (y_pred->shape[0] != y_true->shape[0])
    {
        printf("[ERROR] No compatible shapes between y_true and y_pred.\n");
        exit(1);
    }

    for (int i = 0; i < y_true->shape[0]; i++)
    {
        loss += -1 * log(y_pred->values[i * y_pred->shape[1] + (int)y_true->values[i]]);
    }

    return loss / y_true->shape[0];
}

float nn_accuracy_score(const tensor_t* y_true, const tensor_t* y_pred)
{
    float correct = 0;
    tensor_t* gc = tensor_eq(y_true, y_pred);
   for (int i = 0; i < tensor_numel(y_true); i++)
        correct += gc->values[i];
    return correct / y_true->shape[0];
}
