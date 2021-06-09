X = -3.1:.1:3.1;
T = load('P10.mat')
T1=T.P10(1,:)
plot(X,T1,'+');
title('Training Vectors');
xlabel('Input Vector P');
ylabel('Target Vector T');


eg = 0.00001; 
sc = 2; %spread of the radial basis
net = newrb(X,T1,eg,sc);

plot(X,T1,'+');
xlabel('Input');
X = -3.1:.1:3.1;
Y = net(X);

hold on;
plot(X,Y);
hold off;
legend({'Target','Output'})
pause


