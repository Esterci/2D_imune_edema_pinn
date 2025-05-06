#include "mdf2d.h"

real fb(unsigned i, unsigned j, real Cb, real Cn)
{
    real cb       = 0.154;
    real lambd_nb = 1.8;

    return (cb - lambd_nb * Cn) * Cb;
}

real fn(unsigned i, unsigned j, real Cb, real Cn, real Cn_max)
{
    real mi_n     = 0.2;
    real lambd_bn = 0.1;
    real y_n      = 0.1;

    return (y_n * Cb * (Cn_max - Cn) - lambd_bn * Cn * Cb - mi_n * Cn);
}

real phix(real Cbipj, real Cbimj, real Cbij, unsigned i, unsigned j, unsigned tam_max)
{
    real ax = (Cbipj - Cbimj) / (2 * h);

    if ((i == 0 || i == tam_max - 1) || (j == 0 || j == tam_max - 1) || (ax == 0))
    {
        return 0;
    }
    else if (ax > 0)
    {
        return ((ax * k / h) * (Cbij - Cbimj));
    }
    else if (ax < 0)
    {
        return ((ax * k / h) * (Cbipj - Cbij));
    }
}

real phiy(real Cbijp, real Cbijm, real Cbij, unsigned i, unsigned j, unsigned tam_max)
{
    real ax = (Cbijp - Cbijm) / (2 * h);

    if ((i == 0 || i == tam_max - 1) || (j == 0 || j == tam_max - 1) || (ax == 0))
    {
        return 0;
    }
    else if (ax > 0)
    {
        return ((ax * k / h) * (Cbij - Cbijm));
    }
    else if (ax < 0)
    {
        return ((ax * k / h) * (Cbijp - Cbij));
    }
}

void apply_initial_conditions(real **Cb, unsigned tam_max)
{
    for (unsigned i = 0; i < tam_max; i++)
    {
        for (unsigned j = 0; j < tam_max; j++)
        {
            if ((i * h >= 0.92) && (i * h) <= 1.0)
            {
                if ((j * h >= 0.92) && (j * h) <= 1.0)
                {
                    Cb[i][j] = 0.001;
                }
            }
        }
    }
}

real solve_pde()
{
    real *x;
    real *y;
    real *t;

    unsigned size_x;
    unsigned size_y;
    unsigned size_tt;

    arange_real(&x, 0, 1, h, &size_x);
    arange_real(&y, 0, 1, h, &size_y);
    arange_real(&t, 0, 30, k, &size_tt);

    printf("Size x = %d, y = %d \n ", size_x, size_y);

    real **Cn_old;
    zeros_real(&Cn_old, size_x, size_y);
    real **Cn_new;
    zeros_real(&Cn_new, size_x, size_y);

    real **Cb_old;
    zeros_real(&Cb_old, size_x, size_y);
    real **Cb_new;
    zeros_real(&Cb_new, size_x, size_y);

    // save_result(Cn_old, 1, size_x, size_y);
    // save_result_as_vtk(Cb_old, 0, size_x, size_y, 0);

    printf("Steps in time = %d\nSteps in space_x = %d\nSteps in space_y = %d\n", size_tt, size_x, size_y);

    apply_initial_conditions(Cb_old, size_x);

    unsigned time_step    = size_tt;
    unsigned space_step_x = size_x;
    unsigned space_step_y = size_y;

    unsigned save_number = 0;

    real u_a, u_b, u_c, u_d;
    u_a = u_b = u_c = u_d = 0;

    real r_n, r_b;

    r_n        = (Dn * k) / (phi * (h * h));
    r_b        = (Db * k) / (phi * (h * h));
    real start = omp_get_wtime();
    for (unsigned time = 0; time < time_step; time++)
    {
        for (unsigned i = 0; i < space_step_x; i++)
        {
            for (unsigned j = 0; j < space_step_y; j++)
            {
                real Cb_uijp, Cb_uijm, Cb_uipj, Cb_uimj;
                real Cn_uijp, Cn_uijm, Cn_uipj, Cn_uimj;

                // Tratando Cb
                if (j == size_y - 1)
                    Cb_uijp = 2 * h * u_c + Cb_old[i][size_y - 1];
                else
                    Cb_uijp = Cb_old[i][j + 1];
                //
                if (j == 0)
                    Cb_uijm = 2 * h * u_a + Cb_old[i][1];
                else
                    Cb_uijm = Cb_old[i][j - 1];
                //
                if (i == size_x - 1)
                    Cb_uipj = 2 * h * u_d + Cb_old[size_x - 1][j];
                else
                    Cb_uipj = Cb_old[i + 1][j];
                //
                if (i == 0)
                    Cb_uimj = 2 * h * u_b * Cb_old[1][j];
                else
                    Cb_uimj = Cb_old[i - 1][j];
                //

                // Tratando Cn
                if (j == size_y - 1)
                    Cn_uijp = 2 * h * u_c + Cn_old[i][size_y - 1];
                else
                    Cn_uijp = Cn_old[i][j + 1];
                //
                if (j == 0)
                    Cn_uijm = 2 * h * u_a + Cn_old[i][1];
                else
                    Cn_uijm = Cn_old[i][j - 1];
                //
                if (i == size_x - 1)
                    Cn_uipj = 2 * h * u_d + Cn_old[size_x - 1][j];
                else
                    Cn_uipj = Cn_old[i + 1][j];
                //
                if (i == 0)
                    Cn_uimj = 2 * h * u_b * Cn_old[1][j];
                else
                    Cn_uimj = Cn_old[i - 1][j];

                Cb_new[i][j] =
                    r_b * (Cb_uimj + Cb_uipj - 4 * Cb_old[i][j] + Cb_uijp + Cb_uijm) + (k / phi) * (fb(i, j, Cb_old[i][j], Cn_old[i][j])) + Cb_old[i][j];

                Cn_new[i][j] = r_n * (Cn_uimj + Cn_uipj - 4 * Cn_old[i][j] + Cn_uijp + Cn_uijm) + (k / phi) * (fn(i, j, Cb_old[i][j], Cn_old[i][j], 0.55)) +
                               Cn_old[i][j] + ksi * (-phix(Cb_uipj, Cb_uimj, Cb_old[i][j], i, j, size_x) - phiy(Cb_uijp, Cb_uijm, Cb_old[i][j], i, j, size_y));
            }
        }

        for (int i = 0; i < size_x; ++i)
        {
            for (int j = 0; j < size_y; ++j)
            {
                Cb_old[i][j] = Cb_new[i][j];
                Cn_old[i][j] = Cn_new[i][j];
            }
        }

        if (time % save_rate == 0 || time == (time_step - 1))
        {
            save_result_as_vtk(Cb_old, save_number, size_x, size_y, 1);
            save_result_as_vtk(Cn_old, save_number, size_x, size_y, 0);
            save_number++;
        }
    }
    real end = omp_get_wtime();
    printf("Time to solve the model = %.4e\n", (end - start));
    // printf("terminou");
    free(x);
    free(y);
    free(t);
    for (size_t j = 0; j < size_x; j++)
    {
        free(Cb_new[j]);
    }
    free(Cb_new);
    for (size_t j = 0; j < size_x; j++)
    {
        free(Cb_old[j]);
    }
    free(Cb_old);
    for (size_t j = 0; j < size_x; j++)
    {
        free(Cn_new[j]);
    }
    free(Cn_new);
    for (size_t j = 0; j < size_x; j++)
    {
        free(Cn_old[j]);
    }
    free(Cn_old);
}
