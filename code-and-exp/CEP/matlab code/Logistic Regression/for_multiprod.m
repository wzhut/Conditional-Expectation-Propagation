function [product] = for_multiprod(m1, m2)

os1 = size(m1);
s1 = prod(os1);
s1 = s1 / os1(1) / os1(2);
m1 = reshape(m1, [os1(1), os1(2), s1]);

os2 = size(m2);
s2 = prod(os2);
s2 = s2 / os2(1) / os2(2);
m2 = reshape(m2, [os2(1), os2(2), s2]);

assert(all(os1(3:end) == os2(3:end)));
if os1(1) == os1(2) && os1(1) == 1
    n_s = os2;
elseif os2(1) == os2(2) && os2(1) == 1
    n_s = os1;
    else
    n_s = [os1(1), os2(2), s1];
end
product = zeros(n_s);
for i = 1: s1
    product(:,:,i) = m1(:,:,i) * m2(:,:,i);
end
n_s = [n_s(1:2), os1(3:end)];
product = reshape(product, n_s);

end