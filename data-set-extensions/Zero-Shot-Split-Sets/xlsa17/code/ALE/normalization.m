function [X, m, v, mx] = normalization(X, m, v, mx)

n = size(X, 1);

if(nargin == 1)
	%X1 = X;
	%m = sum(X1) / n;
	m = mean(X);
	%X1 = X1 - repmat(m,[n 1]);
	v = sqrt(sum(bsxfun(@minus, X, m).^2) / n);
	%v = sqrt(sum(X1.^2) / n);	
	idx = find(v == 0);
  v(idx) = 1;
	%X1 = X1 ./ repmat(v,[n 1]);
	X = bsxfun(@rdivide, bsxfun(@minus, X, m), v);
	mx = max(max(X));
	X = X / mx;
else
	X = bsxfun(@rdivide, bsxfun(@minus, X, m), v);
	X = X / mx;
end
