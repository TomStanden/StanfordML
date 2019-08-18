function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pass = find(y ==1);
fail = find(y ~=1);
Xpass = zeros(length(pass),2);
Xfail = zeros(length(fail),2);

%create a for loop that stores the pass scores in one matrix
%and all the fail scores in another

%These loops were actually unnecessary, in fact could have plotted after first 
%two lines of code
for i = 1:length(Xpass)
  Xpass(i,:) = X(pass(i),:);
endfor
for i = 1:length(Xfail)
  Xfail(i,:) = X(fail(i),:);
endfor

plot(Xpass(:,1), Xpass(:,2), 'r+', 'MarkerSize', 8);
plot(Xfail(:,1), Xfail(:,2), 'bo', 'MarkerSize', 8);
xlabel('test 1');
ylabel('test 2');
title('2D plot of students test scores and whether the student got accepted to university')
legend('Accepted', 'Rejected');





% =========================================================================



hold off;

end
