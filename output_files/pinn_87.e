Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x4)

real	0m6.829s
user	0m2.306s
sys	0m1.994s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x4)

real	0m5.678s
user	0m2.328s
sys	0m1.266s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x8)

real	0m4.629s
user	0m2.252s
sys	0m1.067s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x8)

real	0m4.728s
user	0m2.142s
sys	0m0.906s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x16)

real	0m4.374s
user	0m2.100s
sys	0m0.968s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x16)

real	0m5.363s
user	0m2.083s
sys	0m1.052s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x4)

real	0m4.722s
user	0m2.132s
sys	0m1.073s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x4)

real	0m5.468s
user	0m2.151s
sys	0m1.091s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x8)

real	0m4.717s
user	0m2.208s
sys	0m1.197s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x8)

real	0m4.525s
user	0m2.280s
sys	0m1.110s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x16)

real	0m6.584s
user	0m2.243s
sys	0m1.229s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x16)

real	0m4.669s
user	0m2.273s
sys	0m0.956s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x4)

real	0m4.793s
user	0m2.219s
sys	0m0.983s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x4)

real	0m4.863s
user	0m2.389s
sys	0m1.010s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x8)

real	0m4.623s
user	0m2.308s
sys	0m1.049s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x8)

real	0m4.759s
user	0m2.300s
sys	0m1.130s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x16)

real	0m4.526s
user	0m2.384s
sys	0m0.940s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x16)

real	0m4.373s
user	0m2.219s
sys	0m1.039s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x4)

real	0m4.501s
user	0m2.267s
sys	0m0.976s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 255, in <module>
    C_initial_pred = model(t_initial)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/thiago.esterci/.conda/envs/torch_gpu/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x4)

real	0m3.921s
user	0m2.190s
sys	0m0.895s
