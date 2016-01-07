function y=sigmoidscaleup(t,start,finish)

%Input "start" is (t,y) coordinates of start of scale up
%Input "finish" is (t,y) coordiantes of end of scale up
%Input "t" is time

if t>start(1,1)&&t<finish(1,1)
    y=sin(pi*(((t-start(1,1))/...
        (finish(1,1)-start(1,1)))-1/2))*...
        (finish(1,2)-start(1,2))/2+(start(1,2)+finish(1,2))/2;
elseif t<=start(1,1)
    y=start(1,2);
elseif t>=finish(1,1)
    y=finish(1,2);
end

end