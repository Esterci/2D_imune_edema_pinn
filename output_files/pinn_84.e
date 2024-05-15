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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x4)

real	0m6.816s
user	0m2.293s
sys	0m1.884s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x4)

real	0m5.517s
user	0m2.103s
sys	0m1.123s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x8)

real	0m4.529s
user	0m2.080s
sys	0m1.033s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x8)

real	0m4.316s
user	0m2.152s
sys	0m0.791s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x16)

real	0m4.124s
user	0m2.119s
sys	0m0.914s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x16)

real	0m4.985s
user	0m2.064s
sys	0m0.941s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x4)

real	0m5.305s
user	0m2.184s
sys	0m0.964s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x4)

real	0m5.074s
user	0m2.239s
sys	0m1.023s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x8)

real	0m5.101s
user	0m2.231s
sys	0m0.992s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x8)

real	0m4.373s
user	0m2.308s
sys	0m0.966s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x16)

real	0m6.138s
user	0m2.140s
sys	0m1.115s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x16)

real	0m4.952s
user	0m2.131s
sys	0m1.001s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x4)

real	0m3.796s
user	0m2.164s
sys	0m0.763s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x4)

real	0m3.961s
user	0m2.147s
sys	0m0.869s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x8)

real	0m3.905s
user	0m2.239s
sys	0m0.836s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x8)

real	0m3.948s
user	0m2.126s
sys	0m0.935s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x16)

real	0m4.736s
user	0m2.292s
sys	0m1.286s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x16)

real	0m4.717s
user	0m2.172s
sys	0m0.910s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x2 and 4x4)

real	0m4.635s
user	0m2.185s
sys	0m1.022s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 4x4)

real	0m4.153s
user	0m2.087s
sys	0m0.965s
