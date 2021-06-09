load x; load s;
load x1; load s1;

[xn,meanx,stdx,sn,means,stds] = prestd(x,s);
x1n = trastd(x1, meanx, stdx);

mn = min(xn);
mx = max(xn);
net = newff([mn mx],[10 10 1], {'tansig' 'tansig' 'purelin'}, 'traingda');

net.trainParam.epochs = 2000;
net = train(net,xn,sn);

Yn = sim(net,xn);
Y = poststd(Yn, means, stds);
plot(x, Y, x, s)
delta = mean((Y-s).^2);
gtext({'relative error=', num2str(delta)})
