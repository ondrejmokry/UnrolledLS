function data2 = Mat2DataCells(mat)

data2 = (permute(num2cell(mat, [1 2]), [3 4 1 2] ));