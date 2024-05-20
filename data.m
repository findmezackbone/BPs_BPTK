clc
clear

filename = 'D:\1st\BPs_BPTK\Python\pythonresult.csv';
python = csvread(filename);

filename = 'D:\1st\BPs_BPTK\Python\Rresult.csv';
R = csvread(filename);

time = linspace(0,74.995,15000);

rel = abs(python-R)./R;


filename2 = 'D:\1st\BPs_BPTK\Python\pythonresult2.csv';
python2 = csvread(filename2);

filename2 = 'D:\1st\BPs_BPTK\Python\Rresult2.csv';
R2 = csvread(filename2);

time = linspace(0,74.995,15000);

rel2 = abs(python2-R2)./R2;
rel = rel(28:end);
rel2 = rel2(28:end);
%%
semilogy(time(28:end), rel)
xlabel('时间(h)')
ylabel('与前人求解结果的相对误差')
%%
semilogy(time(28:end), rel2)
xlabel('时间(h)')
ylabel('与前人求解结果的相对误差')