#ifndef BMW_H
#define BMW_H
#include <torch/types.h>
#include <torch/extension.h>

torch::Tensor bmw_int8(torch::Tensor A, torch::Tensor W, torch::Tensor alpha);

#endif // BMW_H