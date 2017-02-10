likelihood.map = read.csv(
    file =
    "/home/gdinolov/Research/PDE-solvers/documentation/2-D-advection-diffusion/likelihood-map.csv",
    header = T)

discretization.size = 9;

x = likelihood.map$log.sigma.y[seq(1,discretization.size+1)];
y = x;
z = matrix(byrow = T, data = likelihood.map$neg.ll, nrow = length(x), ncol = length(y));
contour(x,y,z,xlab = "log.sigma.x", ylab = "log.sigma.y");
