clear;
close all hidden;
%    NEWELM  - Inititializes an Elman recurrent network.
%    SIM   - Simulates an Elman recurrent network.
%    TRAIN - Trains an Elman recurrent network.
%    AMPLITUDE DETECTION:
%    Using the above functions an Elman network is trained
%    to output the amplitude of a sine wave being presented
%    in time.
%pause % Strike any key to continue...

%    DEFINING THE PROBLEM
%    ====================
%    The first wave form has an amplitude of 1.
p1 = sin(1:20);     % If input is wave with amplitude of 1
t1 = ones(1,20);    % then output should be 1

%    The second wave form has an amplitude of 2.
p2 = sin(1:20)*2;   % If input is wave with amplitude of 2
t2 = ones(1,20)*2;  % then output should be 2

%    The third wave form has an amplitude of 3.
p3 = sin(1:20)*3;     % If input is wave with amplitude of 3
t3 = ones(1,20)*3;    % then output should be 3

%    The fourth wave form has an amplitude of 4.
p4 = sin(1:20)*4;   % If input is wave with amplitude of 4
t4 = ones(1,20)*4;  % then output should be 4

%    The network will be trained on the sequence formed %	by repeating each wave form twice.
p = [p1 p2 p3 p4 p3 p2 p1 p2 p3 p4 p3 p2 p1 p2 p3 p4 p3 p2 p1 p2 p3 p4 p3 p2 p1];
t = [t1 t2 t3 t4 t3 t2 t1 t2 t3 t4 t3 t2 t1 t2 t3 t4 t3 t2 t1 t2 t3 t4 t3 t2 t1];
figure(1),plot(p);

Pseq = con2seq(p);
Tseq = con2seq(t);
%pause 
% Strike any key to define the Elman network...
%pause % Strike any key to train the Elman network...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFINE THE EHLMAN NETWORK
% =========================
%    The function NEWELM creates an Elman network whose input
% varies from -2 to 2, and has 10 hidden neurons and 1 output.
net = newelm([-2 2],[20 10 1],{'tansig', 'tansig', 'purelin'},'traingdx');
net.trainParam.epochs = 1000;
net.trainParam.show = 5;
net.trainParam.goal = 0.01;

%pause % Strike any key to train the Elman network...

%    TRAINING THE ELMAN NETWORK
%    ===========================
%    TRAIN trains Elman networks.  Here it is trained
%without training paramters. (The default will be used.)
%    Training begins...please wait, this takes a while...tic
[net,tr] = train(net,Pseq,Tseq);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TESTING THE ELMAN NETWORK
% =========================
%    SIM simulates an Elman network for as many timesteps
%    as input vectors in P.
a = sim(net,Pseq);

%    The network outputs and targets can then be plotted.
time = 1:length(p);
figure(3),
plot(time,t,'--',time,cat(2,a{:}))
title('Testing Amplitute Detection')
xlabel('Time Step')
ylabel('Target - -  Output ---')

%    The network does a fairly good job, if not perfect.
%pause 
% Strike any key to test the network's generalization...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    CHECKING GENERALIZATION
%    =======================
%    We will try waveforms with amplitudes of 1.6 and 1.2.
pt1 = sin(1:20)*1.6;     % Input wave with amplitude of 1.6
tt1 = ones(1,20)*1.6;    % We would like the output to be 1.6.

pt2 = sin(1:20)*1.2;     % Try input wave with amplitude of 1.2
tt2 = ones(1,20)*1.2;    % We would like the output to be 1.2.

pt3 = sin(1:20)*1.8;     % Input wave with amplitude of 1.8
tt3 = ones(1,20)*1.8;    % We would like the output to be 1.8.

pt4 = sin(1:20)*0.6;     % Try input wave with amplitude of 0.6
tt4 = ones(1,20)*0.6;    % We would like the output to be 0.6.

%  Repeating each twice results in the series of test inputs.
pg = [pt1 pt2 pt1 pt3 pt1 pt4 pt3 pt4 pt3 pt4 pt2 pt1];
tg = [tt1 tt2 tt1 tt3 tt1 tt4 tt3 tt4 tt3 tt4 tt2 tt1];
pgseq = con2seq(pg);

%pause % Strike any key to see generalization results...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    GENERALIZATION RESULTS
%    ======================
%    SIM is used to simulate the network to these inputs.
a = sim(net,pgseq);

%    The network outputs and targets are plotted.
time = 1:length(pg);
figure(4),
plot(time,tg,'--',time,cat(2,a{:}))
title('Testing Generalization')
xlabel('Time Step')
ylabel('Target - -  Output ---')

%    The network does not do as well for amplitudes for
%    which it was not trained.
%pause % Strike any key for conclusions...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    CONCLUSIONS
%    ===========
%    The Elman recurrent network can learn to recognize time-
%    varying patterns.
%    In this case the network did a fairly good job with only
%  10 neurons in the recurrent layer, and 500 training epochs.
%    More recurrent neurons and longer training times could be
%used to increase the network's accuracy on the training data.
%    Training the network on more amplitudes will result in
%    a network that generalizes better.
%    Type HELP ELMAN for a list of all Elman functions.
% echo off
% End of APPELM1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    ...training done.
figure(5),
semilogy(tr.epoch,tr.perf)
title('Mean Squared Error of Elman Network')
xlabel('Epoch')
ylabel('Mean Squared Error')

%pause % Strike any key to test the Elman network...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
