#ifndef BMM_H
#define BMM_H
#include <torch/types.h>
#include <torch/extension.h>

torch::Tensor bmm_int8(torch::Tensor A, torch::Tensor B, float alpha);

#endif // BMM_H