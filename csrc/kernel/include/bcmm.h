#ifndef BCMM_H
#define BCMM_H
#include <torch/types.h>
#include <torch/extension.h>

torch::Tensor bcmm_int8(torch::Tensor A, torch::Tensor B, float alpha);

#endif // BCMM_H