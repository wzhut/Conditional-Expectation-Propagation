l = zeros(6, 5);
t = lp.ll;
for i = 1 : 6
    for j = 1 :5 
        l(i,j) = t(i,j).mean;
    end
end