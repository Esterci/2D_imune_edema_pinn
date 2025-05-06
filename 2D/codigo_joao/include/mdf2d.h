#ifndef MDF_2D_H
#define MDF_2D_H

#include "defines.h"
#include "msh.h"

#pragma once

real fb(unsigned i, unsigned j, real Cb, real Cn);

real fn(unsigned i, unsigned j, real Cb, real Cn, real Cn_max);

real phix(real Cbipj, real Cbimj, real Cbij, unsigned i, unsigned j, unsigned tam_max);

real phiy(real Cbijp, real Cbijm, real Cbij, unsigned i, unsigned j, unsigned tam_max);

void apply_initial_conditions(real **Cb, unsigned tam_max);

real solve_pde();

#endif