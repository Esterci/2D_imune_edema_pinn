
real	1m18.038s
user	1m10.127s
sys	0m3.531s

real	3m33.322s
user	2m39.529s
sys	0m31.790s

real	1m12.134s
user	1m3.496s
sys	0m6.201s

real	2m2.485s
user	1m57.640s
sys	0m2.108s

real	1m8.113s
user	1m4.773s
sys	0m1.432s

real	2m55.298s
user	2m36.445s
sys	0m13.623s

real	1m1.180s
user	0m57.781s
sys	0m1.669s

real	3m2.223s
user	2m43.993s
sys	0m13.552s

real	0m59.573s
user	0m55.493s
sys	0m2.233s

real	2m52.633s
user	2m30.328s
sys	0m16.920s

real	1m11.668s
user	1m8.544s
sys	0m1.431s

real	2m22.520s
user	2m18.095s
sys	0m1.823s
Traceback (most recent call last):
  File "/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_model.py", line 184, in <module>
    Cp = pk.load(f)
EOFError: Ran out of input

real	0m1.876s
user	0m1.318s
sys	0m0.178s

real	3m2.225s
user	2m43.999s
sys	0m13.961s
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

real	0m3.249s
user	0m2.102s
sys	0m0.598s
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

real	0m3.255s
user	0m2.130s
sys	0m0.627s
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

real	0m3.173s
user	0m2.071s
sys	0m0.596s
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

real	0m3.179s
user	0m2.016s
sys	0m0.597s
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

real	0m3.295s
user	0m2.048s
sys	0m0.665s
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

real	0m3.035s
user	0m1.958s
sys	0m0.594s
