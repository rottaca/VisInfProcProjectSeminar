function [ result, tempBi, tempMo ] = constructSpartialTemporalFilterbank( thetaGarbor, f0, angles, ...
    times, halfGarborRange, s1,s2,uBi1, uBi2, thetaBi1, thetaBi2,uMono, thetaMono )

    result = {};
    
    garbors = constructGarborFilterBank(thetaGarbor,f0,angles,halfGarborRange);
    
    tempBi = constructTemporalBiFilter(thetaBi1,uBi1,s1,thetaBi2,uBi2,s2,times);
    tempMo = constructTemporalMonoFilter(thetaMono,uMono,times);

    for g=garbors
       result(end+1).G = g;
       result(end).combined = combineFilters(g.Ge,g.Go,tempBi,tempMo);
    end
end