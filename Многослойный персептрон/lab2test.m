load x; load s;
load x1; load s1;

[xn,meanx,stdx,sn,means,stds] = prestd(x,s);
x1n = trastd(x1, meanx, stdx);

load mynetff
Yn = sim(net,x1n);
Y = poststd(Yn, means, stds);

labelsx = 1:length(x1);
plot(labelsx, Y, labelsx, s1)
delta = mean((Y-s1).^2);
gtext({'relative error=',num2str(delta)});


