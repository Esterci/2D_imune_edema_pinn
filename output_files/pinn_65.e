
real	1m44.896s
user	1m9.190s
sys	0m17.430s

real	2m38.970s
user	2m29.490s
sys	0m3.008s

real	1m22.105s
user	1m11.475s
sys	0m8.116s

real	2m20.071s
user	2m14.948s
sys	0m2.112s

real	0m43.768s
user	0m40.423s
sys	0m1.682s

real	2m24.791s
user	2m13.043s
sys	0m7.774s

real	1m9.806s
user	0m59.777s
sys	0m7.390s

real	2m4.393s
user	1m59.638s
sys	0m2.064s

real	1m4.212s
user	0m56.316s
sys	0m5.421s

real	2m44.066s
user	2m25.067s
sys	0m14.059s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 184, in <module>
    Cp = pk.load(f)
EOFError: Ran out of input

real	0m1.935s
user	0m1.322s
sys	0m0.179s

real	2m26.351s
user	2m21.575s
sys	0m2.086s

real	0m53.678s
user	0m50.073s
sys	0m1.624s

real	2m28.593s
user	2m23.719s
sys	0m2.212s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x2)

real	0m3.666s
user	0m2.183s
sys	0m0.913s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x2)

real	0m3.671s
user	0m2.088s
sys	0m0.997s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x4 and 8x2)

real	0m3.615s
user	0m2.041s
sys	0m0.986s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x4 and 8x2)

real	0m3.659s
user	0m2.242s
sys	0m0.880s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x2)

real	0m3.769s
user	0m2.194s
sys	0m1.007s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x2)

real	0m3.464s
user	0m2.095s
sys	0m0.874s
