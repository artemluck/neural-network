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
load Plearn; load Tlearn;
load Ptest; load Ttest;

figure(1),plot(Plearn);
figure(2),plot(Ptest);

Pseq = con2seq(Plearn);
Tseq = con2seq(Tlearn);
%pause 
% Strike any key to define the Elman network...
%pause % Strike any key to train the Elman network...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFINE THE EHLMAN NETWORK
% =========================
%    The function NEWELM creates an Elman network whose input
% varies from -2 to 2, and has 10 hidden neurons and 1 output.
net = newelm([-2 2],[10 1],{'tansig' 'purelin'},'traingdx');
net.trainParam.epochs = 700;
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
time = 1:length(Plearn);
figure(3),
plot(time,Tlearn,'--',time,cat(2,a{:}))
title('Testing Amplitute Detection')
xlabel('Time Step')
ylabel('Target - -  Output ---')

%    The network does a fairly good job, if not perfect.
%pause 
% Strike any key to test the network's generalization...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    CHECKING GENERALIZATION
%    =======================

pgseq = con2seq(Ptest);
tg = Ttest;

%pause % Strike any key to see generalization results...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    GENERALIZATION RESULTS
%    ======================
%    SIM is used to simulate the network to these inputs.
a = sim(net,pgseq);

%    The network outputs and targets are plotted.
time = 1:length(Ptest);
figure(4),
plot(time,Ttest,'--',time,cat(2,a{:}))
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
