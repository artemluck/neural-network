load x; load s;
load x1; load s1;

[xn,meanx,stdx,sn,means,stds] = prestd(x,s);
x1n = trastd(x1, meanx, stdx);

mn = min(xn);
mx = max(xn);
net = newff([mn mx],[10 10 1], {'tansig' 'tansig' 'purelin'}, 'traingda');

net.trainParam.epochs = 10;
net = train(net,xn,sn);

Yn = sim(net,x1n);
Y = poststd(Yn, means, stds);
labelsx = 1:length(x1);
plot(labelsx, Y, labelsx, s1)
save mynetff mynetff
