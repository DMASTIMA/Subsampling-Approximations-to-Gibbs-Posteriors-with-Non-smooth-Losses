import numpy as np
from scipy import stats
from scipy.linalg import solve
import time
import warnings

warnings.filterwarnings('ignore')




def huber_loss_grad(w, X, Y):

    delta = 2
    residuals = Y - X @ w
    abs_r = np.abs(residuals)

    # loss
    quadratic = np.minimum(abs_r, delta)
    losses = 0.5 * quadratic ** 2 + delta * (abs_r - quadratic)

    # gradient
    grad_res = np.where(abs_r <= delta, residuals, delta * np.sign(residuals))
    grads = -X * grad_res[:, np.newaxis]

    return losses, grads

    # Fulll
def lel(w, X, Y):
    losses , grads = huber_loss_grad(w, X, Y)
    return -np.sum(losses), np.sum(grads, axis=0)

    #Sub_0
def lel0cv(w, X1, Y1, H2):
    losses ,grads = huber_loss_grad(w, X1, Y1)

    loss_sum = np.sum(losses)
    H2_sum = np.sum(H2)
    g = loss_sum - H2_sum

    return -g, np.sum(grads, axis=0)

    #Sub_1
def lel1cv(w, X1, Y1, H2, Gst, H3, thetahat):
    losses, grads = huber_loss_grad(w, X1, Y1)  # shape=(n1, d)
    loss_sum = np.sum(losses)
    H2_sum = np.sum(H2)
    grads_sum = np.sum(grads, axis=0)

    Gst1 = np.sum(H3, axis=0)
    expr = loss_sum - H2_sum - Gst1 @ (w - thetahat)
    grad_w = (grads_sum - Gst1) * (expr > 0)

    g = np.maximum(0, expr) + Gst @ (w - thetahat)

    h = grad_w + Gst
    return -g, h


def generate_data(dd, n):
    # Generate data
    X = np.random.multivariate_normal(np.zeros(dd), np.eye(dd), n)
    err_normal = np.random.normal(0, 1, int(4 * n / 5))
    err_t = stats.t.rvs(4, size=int(n / 5))
    err = np.concatenate([err_normal, err_t])
    Y = X @ np.ones(dd) + err
    return X, Y


def sample_full(X, Y, dd, n, KK):
    start_time = time.time()
    Theta = np.zeros((KK, dd))

    # Initialize
    thetahat = np.linalg.lstsq(X, Y, rcond=None)[0]
    Theta[0] = thetahat

    q1, dir1 = lel(Theta[0], X, Y)
    sigma = 0.03 * np.sqrt(3000) / (np.sqrt(n) * dd ** (1 / 6))

    for l in range(1, KK):
        # Propose new sample
        ss = np.random.multivariate_normal(
            Theta[l - 1] - sigma ** 2 / 2 * dir1,
            sigma ** 2 * np.eye(dd)
        )

        q2, dir2 = lel(ss, X, Y)

        # Calculate acceptance probability
        if (all(-10 <= s <= 10 for s in ss)):
            log_t = (q2 - q1 -
                     1 / (2 * sigma ** 2) * (
                             np.dot(Theta[l - 1] - ss + sigma ** 2 / 2 * dir2,
                                    Theta[l - 1] - ss + sigma ** 2 / 2 * dir2) -
                             np.dot(ss - Theta[l - 1] + sigma ** 2 / 2 * dir1,
                                    ss - Theta[l - 1] + sigma ** 2 / 2 * dir1)
                     ))
            t = np.exp(log_t)
        else:
            t = 0

        # Metropolis-Hastings accept/reject
        if t > 1:
            Theta[l] = ss
            q1 = q2
            dir1 = dir2
        else:
            u = np.random.uniform(0, 1)
            if u < t:
                Theta[l] = ss
                q1 = q2
                dir1 = dir2
            else:
                Theta[l] = Theta[l - 1]

    end_time = time.time()
    time_taken = end_time - start_time

    return Theta, time_taken


def sample_sub(X, Y, dd, n, m, KK, order, replace=False):
    start_time = time.time()

    thetahat = np.linalg.lstsq(X, Y, rcond=None)[0]
    GH , HH = huber_loss_grad(thetahat, X, Y)


    Theta = np.zeros((KK, dd))
    Theta[0] = thetahat

    S0 = np.random.choice(n, m, replace=replace)

    if order == 0:
        alpha = m ** (-1/3)
        q1, dir1 = lel0cv(Theta[0], X[S0], Y[S0], GH[S0])
    else:
        # Calculate Hessian-related terms
        factor = 1
        alpha = n / m * factor
        Gst = np.sum(HH, axis=0)
        Gst = Gst / alpha
        q1, dir1 = lel1cv(Theta[0], X[S0], Y[S0], GH[S0], Gst, HH[S0], thetahat)

    dir1 = alpha * dir1
    sigma = 0.03 * np.sqrt(3000) / (np.sqrt(alpha * m ) * dd ** (1 / 6))

    for l in range(1, KK):
        ss = np.random.multivariate_normal(
            Theta[l - 1] - sigma ** 2 / 2 * dir1 ,
            sigma ** 2 * np.eye(dd)
        )

        u1 = np.random.uniform(0, 1)
        if u1 < 0.5 :
            S1 = S0
        else:
            S1 = np.random.choice(n, m, replace=replace)

        if order == 0:
            q2, dir2 = lel0cv(ss, X[S1], Y[S1], GH[S1])
        else:
            q2, dir2 = lel1cv(ss, X[S1], Y[S1], GH[S1], Gst, HH[S1], thetahat)

        dir2 = alpha * dir2

        if (all(-10 <= s <= 10 for s in ss)):
            log_t = (alpha * q2 - alpha* q1 -
                     1 / (2 * sigma ** 2) * (
                             np.dot(Theta[l - 1] - ss + sigma ** 2 / 2 * dir2  ,
                                    Theta[l - 1] - ss + sigma ** 2 / 2 * dir2 ) -
                             np.dot(ss - Theta[l - 1] + sigma ** 2 / 2 * dir1 ,
                                    ss - Theta[l - 1] + sigma ** 2 / 2 * dir1 )
                     ))
            t = np.exp(log_t)
        else:
            t = 0

        if t > 1:
            Theta[l] = ss
            q1 = q2
            dir1 = dir2
            S0 = S1
        else:
            u = np.random.uniform(0, 1)
            if u < t:
                Theta[l] = ss
                q1 = q2
                dir1 = dir2
                S0 = S1
            else:
                Theta[l] = Theta[l - 1]

    end_time = time.time()
    time_taken = end_time - start_time

    if order == 0:
        thetahat1 = np.mean(Theta, axis=0)
        if order == 0:
            thetahat1 = np.mean(Theta, axis=0)
            Theta = (Theta - thetahat1) / np.sqrt( 1 / alpha) * np.sqrt(m / n) + thetahat1
    else:
        thetahat1 = np.mean(Theta, axis=0)

    return Theta, thetahat1, time_taken
