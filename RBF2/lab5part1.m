X = -2.1:.1:2.3;
T = load('P11.mat')
T1=T.P11(1,:)
plot(X,T1,'+');
title('Training Vectors');
xlabel('Input Vector P');
ylabel('Target Vector T');

pause
eg = 0.001; 
sc = 0.000001; %spread of the radial basis
net = newrb(X,T1,eg,sc);

plot(X,T1,'+');
xlabel('Input');
Y = net(X);

hold on;
plot(X,Y);
hold off;
legend({'Target','Output'})
pause


