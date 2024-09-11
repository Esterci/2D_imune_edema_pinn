#ifndef MSH_H
#define MSH_H

#include "defines.h"

#pragma once

#define h   0.003333333
#define k   0.0001

#define Db  0.0001                        // termo da difusão da bactéria
#define Dn  0.0001                        // termo da difusão do neutrófilo
#define phi 0.2                           // termo phi do sistema de EDP
#define ksi 0.0                           // taxa da quimiotaxia
#define CFL ((Db * k) / ((2 * (h * h))))  // condição de CFL menor do que 1

void save_result_as_vtk(real **concentration_matrix, unsigned step, unsigned size_x, unsigned size_y, unsigned indent);
void save_result(real **concentration_matrix, unsigned step, unsigned size_x, unsigned size_y);
void zeros_real(real ***vector, unsigned size_x, unsigned size_y);
void arange_real(real **vector, real start, real stop, real step, unsigned *size);

#endif
