


theta = [0, 0]


x = [0, 1, 2, 3]
y = [1, 3, 5, 7]

def grad(theta):
    t0 = 0;
    t1 = 0;
    for i in range(len(x)):
        t0 += ( y[i] - ( theta[0] * x[i] + theta[1] ) ) * x[i]
        t1 += ( y[i] - ( theta[0] * x[i] + theta[1] ) )
    return [t0/len(x), t1/len(x)]

for i in range(10000):
    new_theta = grad(theta)
    if new_theta[0] is 0:
        break
    theta[0] += 0.01 * new_theta[0]
    theta[1] += 0.01 * new_theta[1]
print theta[0], theta[1]
