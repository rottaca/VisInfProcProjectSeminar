function [ combined ] = combineFilters( Geven,Godd, tBi,tMono )

    tRange = length(tBi);
    gSz = size(Geven,1);
   
    combined = repmat(reshape(tBi,1,1,[]),gSz,gSz,1).*repmat(Geven,1,1,tRange) ...
             + repmat(reshape(tMono,1,1,[]),gSz,gSz,1).*repmat(Godd,1,1,tRange);

end

