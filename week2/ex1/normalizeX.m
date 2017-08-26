function testNorm = normalizeX(X, sigma, mu)

	testNorm = X;
	k = size(X)
	m = k(1,2)


	for i = 1:m

		testNorm(:,i) = (X(:,i) - mu(:,i)) ./ sigma(:,i)
	end
end
