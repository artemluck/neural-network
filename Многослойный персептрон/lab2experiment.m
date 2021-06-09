load x; load s;
load x1; load s1;

[xn,meanx,stdx,sn,means,stds] = prestd(x,s);
x1n = trastd(x1, meanx, stdx);

mn = min(xn);
mx = max(xn);

MAX_N = 31;
layers = 1:MAX_N;
delta1 = 1:MAX_N;
delta=1:MAX_N;

for i=1:MAX_N
    l = 10+i-9;
    layers(i) = l;
    net = newff([mn mx],[l l 1], {'tansig' 'tansig' 'purelin'}, 'traingda');

    net.trainParam.epochs = 2000;
    net = train(net,xn,sn);

    Y1n = sim(net,x1n);
    Y1 = poststd(Y1n, means, stds);
    Yn = sim(net, xn);
    Y = poststd(Yn, means, stds);
    
    delta(i) = mean((Y-s).^2);
    delta1(i) = mean((Y1-s1).^2);
end

plot(layers, delta, layers, delta1)
