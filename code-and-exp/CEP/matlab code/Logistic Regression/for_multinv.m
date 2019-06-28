function [invs] = for_multinv(org)

os = size(org);
s = prod(os);
s = s / os(1)^2;
org = reshape(org, [os(1), os(1), s]);
invs = zeros(size(org));

for i = 1: s
    if cond(org(:,:,i)) > 1e15
        disp('singular');
    end
    invs(:,:,i) = inv(org(:,:,i));
end
invs = reshape(invs, os);

end