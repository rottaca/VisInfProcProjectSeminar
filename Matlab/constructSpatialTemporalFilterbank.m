function [ result, tempBi, tempMo ] = constructSpatialTemporalFilterbank( sigmaGabor, f0, angles, ...
    times, halfGaborRange,resolutionSpatial, s1,s2,uBi1, uBi2, sigmaBi1, sigmaBi2,uMono, sigmaMono )

    result = {};
    
    garbors = constructGaborFilterBank(sigmaGabor,f0,angles,halfGaborRange,resolutionSpatial);
    
    tempBi = constructTemporalBiFilter(sigmaBi1,uBi1,s1,sigmaBi2,uBi2,s2,times);
    tempMo = constructTemporalMonoFilter(sigmaMono,uMono,times);

    for g=garbors
       result(end+1).G = g;
       result(end).combined = combineFilters(g.Ge,g.Go,tempBi,tempMo);
    end
end