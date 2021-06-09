function lab1()    
    hF1=figure('Position', [500 500 100 100]);
    hP=uicontrol(hF1,...
        'Style','pushbutton',...
        'String', 'Perseptron', ...
        'Position', [10, 60, 100, 25],...
        'Callback', @NNPerseptron);

    hMLP=uicontrol(hF1, ...
        'Style','pushbutton', ...
        'String', 'MLP', ...
        'Position', [10 35 100 25],...
        'Callback', @NNMLP); 
    
    hL=uicontrol(hF1,...
        'Style','pushbutton',...
        'String', 'Show Labels', ...
        'Position', [10, 10, 100, 25],...
        'Callback', @show_labels);
end


function NNPerseptron(src,event)
    net = newp([0 1; 0 1; 0 1],1);
    
    net=train_my_net(net);
    test_my_net(net);
end

function NNMLP(src,event)    
    net = newff([0 1; 0 1; 0 1],[3 1], {'tansig' 'logsig'});
    
    net=train_my_net(net);
    test_my_net(net);
end


function n=train_my_net(net)
    P = [0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1];
    T = [0 0 1 0 0 1 0 0];
    
    Ybefore_train = sim(net,P)
    net.trainParam.epochs = 50;
    net = train(net,P,T);
    Yafter_train = round(double(sim(net,P)))
    
    figure('Name', 'Outputs')
    plotpv(P, Yafter_train);
    figure('Name', 'Dividing surface')
    plotpc(net.IW{1}, net.b{1});
    n=net;
end


function test_my_net(net)
    P = [0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1];
    P_noised = P;
    T = [0 0 1 0 0 1 0 0];
    
    for i=1:3
        for j=1:8
            noise=normrnd(0,0.1);
            P_noised(i,j) = P(i,j) + noise;
        end
    end
    
    Ytest = sim(net,P_noised)
    figure('Name', 'Test Outputs')
    plotpv(P, round(double(Ytest)))
    MSE = mean((Ytest-T).^2)
end


function show_labels(src,event)
    P = [0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1];
    T = [0 0 1 0 0 1 0 0];
    
    figure('Name', 'Labels');
    plotpv(P, T);
end

