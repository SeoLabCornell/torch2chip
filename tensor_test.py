import torch

def main():
    x1 = torch.load("/share/seo/t2c_ckpt/int8/llama3.2-3b/tensors/model.layers.0.mlp.down_proj.ops_x1.pt")
    x2 = torch.load("/share/seo/t2c_ckpt/int8/llama3.2-3b/tensors/model.layers.0.mlp.down_proj.ops_x2.pt")
    y = torch.load("/share/seo/t2c_ckpt/int8/llama3.2-3b/tensors/model.layers.0.mlp.down_proj.ops_y.pt")

    print(x1.unique())
    print(x2.unique())
    print(y.unique())

if __name__ == "__main__":
    main()