#ifndef QBMW_H
#define QBMW_H
#include <torch/types.h>
#include <torch/extension.h>

torch::Tensor qbmw(torch::Tensor A, torch::Tensor W);

torch::Tensor qbmw_scaled(torch::Tensor A, torch::Tensor W, torch::Tensor scale);

torch::Tensor qbmw_quantized(torch::Tensor A, torch::Tensor W, torch::Tensor scale);

#endif // BMW_H