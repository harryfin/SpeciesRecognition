function mappedL = mapLabel(L, classes)
    mappedL = -1 * ones(size(L));
    for i=1:length(classes)
        idx = find(L == classes(i));
        mappedL(idx) = i;
    end
end