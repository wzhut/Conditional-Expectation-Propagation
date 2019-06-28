%x is N by 1, y is [m1, m2, N]
function z =mtimesx_scale(x, y)
    [m1, m2, N] = size(y);
    z = repmat(reshape(x,[1,1,N]), [m1,m2]).*y;
end