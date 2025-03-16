#include "include/bmm.h"
#include "include/bmw.h"
#include "include/qbmw.h"
#include "include/bcmm.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Binding the function to Python
PYBIND11_MODULE(t2c_gemm, m) {
    // Function wrapper
    m.def("bmm_int8", &bmm_int8,
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("alpha"),
          R"pbdoc(
            Batched matrix multiplication with int8 inputs.

            Args:
                A (torch.Tensor): Tensor of type int8 (3-D).
                B (torch.Tensor): Tensor of type int8 (3-D).
                alpha (float): Scalar value for scaling the result.

            Returns:
                torch.Tensor: Result of the batched matrix multiplication.
          )pbdoc");

    m.def("bmw_int8", &bmw_int8,
          pybind11::arg("A"), pybind11::arg("W"), pybind11::arg("alpha"),
          R"pbdoc(
            Batched matrix multiplication with int8 inputs.

            Args:
                A (torch.Tensor): Input Activation Tensor of type int8 (3-D).
                W (torch.Tensor): Weight Tensor of type int8 (2-D).
                alpha (float): Scalar value for scaling the result.

            Returns:
                torch.Tensor: Result of the batched matrix multiplication.
          )pbdoc");

    m.def("qbmw", &qbmw,
          pybind11::arg("A"), pybind11::arg("W"),
          R"pbdoc(
            Batched matrix multiplication with int8 inputs.

            Args:
                A (torch.Tensor): Input Activation Tensor of type int8 (3-D).
                W (torch.Tensor): Weight Tensor of type int8 (2-D).

            Returns:
                torch.Tensor: Result of the batched matrix multiplication.
          )pbdoc");

    m.def("qbmw_scaled", &qbmw_scaled,
          pybind11::arg("A"), pybind11::arg("W"), pybind11::arg("scale"), 
          R"pbdoc(
            Batched matrix multiplication with int8 inputs.

            Args:
                A (torch.Tensor): Input Activation Tensor of type int8 (3-D).
                W (torch.Tensor): Weight Tensor of type int8 (2-D).
                scale (torch.Tensor): Scaling factor.

            Returns:
                torch.Tensor: Result of the batched matrix multiplication.
          )pbdoc");

    m.def("qbmw_quantized", &qbmw_quantized,
          pybind11::arg("A"), pybind11::arg("W"), pybind11::arg("scale"), 
          R"pbdoc(
            Batched matrix multiplication with int8 inputs.

            Args:
                A (torch.Tensor): Input Activation Tensor of type int8 (3-D).
                W (torch.Tensor): Weight Tensor of type int8 (2-D).
                scale (torch.Tensor): Scaling factor.

            Returns:
                torch.Tensor: Result of the batched matrix multiplication.
          )pbdoc");

    m.def("bcmm_int8", &bcmm_int8,
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("alpha"),
          R"pbdoc(
            Batched matrix multiplication with int8 inputs.

            Args:
                A (torch.Tensor): Tensor of type int8 (4-D).
                B (torch.Tensor): Tensor of type int8 (4-D).
                alpha (float): Scalar value for scaling the result.

            Returns:
                torch.Tensor: Result of the batched matrix multiplication.
          )pbdoc");
}