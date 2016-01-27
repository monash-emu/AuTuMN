%Get rid of NaNs
for b=1:size(time,1)-1
    if isnan(mdrpropinc(b,1))==1
        mdrpropinc(b,1)=mdrpropinc(b+1,1);
    end
    if isnan(inc(b,1))==1
        inc(b,1)=inc(b+1,1);
    end
    if isnan(mdrpropretreat(b,1))==1
        mdrpropretreat(b,1)=mdrpropretreat(b+1,1);
    end
    if isnan(mort(b,1))==1
        mort(b,1)=mort(b+1,1);
    end
end