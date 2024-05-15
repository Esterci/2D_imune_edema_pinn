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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x16)

real	0m6.716s
user	0m2.188s
sys	0m1.921s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x16)

real	0m5.486s
user	0m2.116s
sys	0m1.128s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x4)

real	0m4.831s
user	0m2.196s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x4)

real	0m4.726s
user	0m2.111s
sys	0m0.927s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x8)

real	0m4.373s
user	0m2.105s
sys	0m0.901s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x8)

real	0m5.539s
user	0m2.065s
sys	0m0.989s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x16)

real	0m4.561s
user	0m2.191s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x16)

real	0m5.332s
user	0m2.339s
sys	0m1.024s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x4)

real	0m4.716s
user	0m2.197s
sys	0m1.068s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x4)

real	0m4.464s
user	0m2.319s
sys	0m0.952s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x8)

real	0m6.214s
user	0m2.105s
sys	0m1.065s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x8)

real	0m4.974s
user	0m2.147s
sys	0m1.005s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x16)

real	0m4.398s
user	0m2.139s
sys	0m0.902s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x16)

real	0m4.138s
user	0m2.178s
sys	0m0.927s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x4)

real	0m3.794s
user	0m2.203s
sys	0m0.870s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x4)

real	0m4.321s
user	0m2.212s
sys	0m0.930s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x8)

real	0m3.617s
user	0m2.127s
sys	0m0.699s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x8)

real	0m5.084s
user	0m2.083s
sys	0m1.045s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (10000x8 and 16x16)

real	0m4.479s
user	0m2.210s
sys	0m1.054s
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
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x8 and 16x16)

real	0m4.107s
user	0m2.053s
sys	0m0.907s
