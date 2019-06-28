%p: the specified coefficients
%w: the real ipnut for the function to integrate
function [y] = func_quad_v2(w, p1, p2, s1, s2)
    t1 = normpdf(w,zeros(size(w)),ones(size(w))*sqrt(s1));
    t2 = normpdf(w,zeros(size(w)),ones(size(w))*sqrt(s2));
    y = (p1*t1 + p2*t2)./(t1+t2);
end