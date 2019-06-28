large_matrix = rand([10,10,200000]);
large_matrix1 = rand([10,10,200000]);
tic;
mi = multiprod(large_matrix,large_matrix1);
toc;

tic;
% for i = 1:20000
    mi1 = my_multiprod(large_matrix, large_matrix1);
% end
toc;