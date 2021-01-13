#ifndef _NN_H_
#define _NN_H_

#include "tensor.h"

tensor_t* nn_relu(tensor_t* t);
tensor_t* nn_softmax(tensor_t* t, uint32_t axis);

float nn_sparse_ce_loss(const tensor_t* y_true, const tensor_t* y_pred);

float nn_accuracy_score(const tensor_t* y_true, const tensor_t* y_pred);

#endif
