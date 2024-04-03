#include "msh.h"

void save_result_as_vtk(real **concentration_matrix, unsigned step, unsigned size_x, unsigned size_y, unsigned indent)
{
    FILE *vtk_file;
    char  filename[30];
    if (indent == 1)
        sprintf(filename, "../cb_over_time/concentration%d.vtk", step);
    if (indent == 0)
        sprintf(filename, "../cn_over_time/concentration%d.vtk", step);

    vtk_file = fopen(filename, "w");

    fprintf(vtk_file, "# vtk DataFile Version 3.0\n");
    fprintf(vtk_file, "results.vtk\n");
    fprintf(vtk_file, "ASCII\n");
    fprintf(vtk_file, "DATASET RECTILINEAR_GRID\n");
    fprintf(vtk_file, "DIMENSIONS %d %d 1\n", size_x, size_y);

    fprintf(vtk_file, "X_COORDINATES %d double\n", size_x);
    for (int i = 0; i < size_x; i++)
    {
        fprintf(vtk_file, "%f ", i * (h / 1));
    }
    fprintf(vtk_file, "\n");

    fprintf(vtk_file, "Y_COORDINATES %d double\n", size_y);
    for (int j = 0; j < size_y; j++)
    {
        fprintf(vtk_file, "%f ", j * (h / 1));
    }
    fprintf(vtk_file, "\n");

    fprintf(vtk_file, "Z_COORDINATES 1 double\n");
    fprintf(vtk_file, "0");
    fprintf(vtk_file, "\n");

    fprintf(vtk_file, "POINT_DATA %d \n", size_y * size_x * 1);
    fprintf(vtk_file, "FIELD FieldData 1 \n");
    fprintf(vtk_file, "Concentração 1 %d double \n", size_y * size_x * 1);
    for (int j = 0; j < size_x; j++)
    {
        for (int i = 0; i < size_y; i++)
        {
            fprintf(vtk_file, "%f \n", concentration_matrix[j][i]);
        }
    }
    fprintf(vtk_file, "\n");
    fclose(vtk_file);
}

void save_result(real **concentration_matrix, unsigned step, unsigned size_x, unsigned size_y)
{
    FILE *plot_file;
    char  filename[30];
    sprintf(filename, "../concentration_final.txt");
    plot_file = fopen(filename, "w");

    fprintf(plot_file, "x\t\ty\t\tconcentracao\n");

    for (int i = 0; i < size_x; ++i)
    {
        for (int j = 0; j < size_y; ++j)
        {
            fprintf(plot_file, "%f\t\t%f\t\t%f\n", i * h, j * h, concentration_matrix[i][j]);
        }
    }
}

void zeros_real(real ***vector, unsigned size_x, unsigned size_y)
{
    *vector = (real **)malloc((size_x) * sizeof(real *));
    for (size_t j = 0; j < size_x; j++)
    {
        (*vector)[j] = (real *)malloc((size_y) * sizeof(real));
    }

    real prx = 0;

    for (size_t i = 0; i < size_x; i++)
    {
        for (size_t j = 0; j < size_y; j++)
        {
            (*vector)[i][j] = prx;
        }
    }
}

void arange_real(real **vector, real start, real stop, real step, unsigned *size)
{
    (*size) = ((stop - start) / (step)) + 1;

    *vector = (real *)malloc((*size) * sizeof(real));

    real     prx = 0;
    unsigned i   = 0;
    while (prx < stop + (step / 2))
    {
        (*vector)[i] = prx;
        prx += step;
        i++;
    }
}