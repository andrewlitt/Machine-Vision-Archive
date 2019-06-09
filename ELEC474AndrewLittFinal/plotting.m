fileID = fopen('output.txt');
data = textscan(fileID,'%f %f %f','Delimiter',',');
X = data{1};
Y = data{2};
Z = data{3};
scatter3(X,Y,Z);