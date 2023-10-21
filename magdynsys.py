import numpy as np


def Larmor_center(X, b):
    '''
    A particle in a uniform magnetic field B=(0,0,b) for b>0
    travels along (Larmor) circles determined by the initial
    position, velocity and the field strength b.

    This function determines the cartesian coords for
    the center (L1,L2) for given intial conditions and field
    strength.

    Input:
    X - [x1 x2 v1 v2], a vector of a position x1 x2 and
        velocity v1 v2 of a particle in cartesian coords
    b - magnetic field strength

    Output:
    [L1 L2]  - the center of the Larmor circle in cartesian coords
    '''

    x1, x2, v1, v2 = X
    
    return np.array([x1+v2/b,x2-v1/b])


def reflect_mat(C):
    '''
    Given two circles, consider the line through their centers.
    The two centers are invariant under reflections in this line.
    This is a function that computes the reflection in a line
    through the origin and another point.

    Input:
    C - [C1 C2], the position in cartesian coords of the second point

    OUtput:
    2x2 matrix representing the reflection in the line joining C
    and the origin
    '''

    C1, C2 = C
    A = C1**2 - C2**2
    B = 2*C1*C2
    
    return np.array([[A,B],[B,-A]]) / np.sum(C**2)


def SInToSOut(X,b):
    '''
    A particle travels in a circular region S in the plane
    under the influence of a uniform magnetic field B=(0,0,b)
    for b>0. Given a point and transversal velocity on the
    boundary of S, we can compute the exit position and
    velocity using reflections.

    Input:
    X - [x1 x2 v1 v2], position x1 x2 and velociy v1 v2 in
        cartesian coordinates of a particle entering the
        circular magnetic region.
    b - the magnetic field strength of the region.
    '''

    pos, vel = X[:2], X[2:]
    shiftX = pos - 1/2
    A = reflect_mat(Larmor_center(np.block([shiftX,vel]),b))
    new_pos = A @ shiftX + 1/2
    new_vel = - A @ vel
    
    return np.block([new_pos,new_vel])


def circle_and_line(X, R):
    '''
    A function for computing the point of intersection of
    a line and a circle centered at (1/2, 1/2) given a point 
    and a vector on the line and the radius of the circle.

    A line in the plane can be parametrized by t as

    (x+vt, y+wt),

    to find the intersection of this line with a circle

    (x-1/2)^2 + (y-1/2)^2 = R^2,

    we plug in the parametrization of the line into
    the equation of the circle and solve a quadratic
    equation in t. Plugging the value of t back into the
    equation of the line gives us the point.

    Input:
    X - [x1 x2 v1 v2], a point x1 x2 and vector v1 v2 in
        cartesian coords on a line
    R - the radius of the circle centered at (1/2,1/2)

    Output:
    Y - [y1 y2 w1 w2] the point and velocity at which the
        line intersects the circle, if there is no
        intersection y1 y2 are set to np.Inf
    '''

    # the coefficients of the quadratic equation in t
    A = np.sum(X[2:]**2)
    B = np.dot(X[:2]-1/2,X[2:])
    C = np.sum((X[:2]-1/2)**2)-R**2

    # simplified expression for the descriminant of the
    # quadratic equation
    d = B**2-C*A

    # if the descriminant is positive then we have collision
    # we take the t that is smaller, so the first collision
    if d>=0:
        t = (-B - np.sqrt(d))/A
        return np.block([X[2:]*t+X[:2],X[2:]])

    return np.block([np.ones(2)*np.Inf,X[2:]])


def cellToCell(X, tol=1e-15):
    '''
    takes a point and maps it to a point on the boundary

    x=0 or x=1 for y in [0,1]
    y=0 or y=1 for x in [0,1]

    while also translating the point to the right side of the square
    depending on the direction of the velocity, this accounts for the fact
    the motion is on the 2-torus

    Input:
    X - [x1 x2 v1 v2], a point x1 x2 and vector v1 v2 in
        cartesian coords on a line.
    tol - tolerance for the computation used to decide
          when to consider a line to be parallel to the
          4 lines. If either v1 or v2 are smaller than the
          tolerance, it is assumed that the line is parallel
          to x=0 and x=1 or y=0 and y=1, respectively

    Output:
    Y - [y1 y2 w1 w2] the point and velocity at which the
        line intersects one of the 4 lines but wrapped
        to account for motion on the 2-torus.
    [s1 s2] - a vector indicating to relative square of the
              2-torus into which the trajectory is entering
    '''

    x1, x2, v1, v2 = X
    t1, t2 = np.Inf, np.Inf
    d1, d2 = int(v1 > 0), int(v2 > 0)
    s1, s2 = 0, 0

    # checking for parallel velocities
    if np.abs(v2) > tol:
        s2 = np.sign(v2)
        t2 = (d2 - x2)/v2

    if np.abs(v1) > tol:
        s1 = np.sign(v1)
        t1 = (d1 - x1)/v1

    # picking the first intersection time t
    if np.abs(t1 - t2) < tol:
        # we wrap the position to the other side of the square for convenience
        return np.array([1 - d1, 1 - d2, v1, v2]),     np.array([s1, s2])
    elif t1 < t2:
        return np.array([1 - d1, v2*t1 + x2, v1, v2]), np.array([s1,  0])
    else:
        return np.array([v1*t2 + x1, 1 - d2, v1, v2]), np.array([ 0, s2])


def SOutToSIn(X, R, maxIt=1000, tol=1e-15):
    '''
    Function that maps a point on the circle, centered at (1/2,1/2) 
    with radius R, with outward pointing velocity to a point on the
    circle pointing inward. The motion is linear and is wrapped to 
    the same square [0,1]^2.
    
    Input:
    X - [x1 x2 v1 v2], a point x1 x2 and vector v1 v2 in
        cartesian coords on a line.
    R - radius of the circle
    maxIt - number of times we can wrap the through the square [0,1]^2
            before we give up trying to find the next point.
    tol - tolerance for the cellToCell function
    
    Output:
    Xn - the new values for [x1 x2 v1 v2], x1 x2 will be on the circle
         v1 v2 will point inward
    xydir - an array of directions. Wrapping the square can be viewed
            as a translation, each time we translate we record
            the opposite of the translation directions 
    entered - True if we did not exceed maxI.t
    '''
    
    xydir = np.zeros((maxIt, 2))
    Xn, xydir[0,:] = cellToCell(X, tol)
    entered = 0

    for m in range(1, maxIt):
        circ = circle_and_line(Xn, R)
        if circ[0] != np.Inf:
            Xn = circ
            entered = 1
            break
        Xn, xydir[m,:] = cellToCell(Xn, tol)
    
    return Xn, xydir[:m,:], entered


def orbit(X, R, b, n=1000, maxIt=1000, tol=1e-15):
    '''
    For initial conditions X, a point on the circle, centered 
    at (1/2,1/2) with radius R, with outward pointing velocity,
    we compute the first n times the trajectory returns to a
    point on the circle with outward pointing velocity. The
    mangetic disc has a field B=(0,0,b) with strength b > 0.
    
    Input:
    X - [x1 x2 v1 v2], a point x1 x2 and vector v1 v2 in
        cartesian coords on a line.
    R - radius of the magnetic disc.
    n - number of steps to compute for the trajectory.
    maxIt - max number of iterations to pass to SOutToSIn.
    tol - tolerance for the cell2Cell function.

    Output:
    XOut - array of outward pointing points that we visit 
           for each step.
    XIn - array of inward pointing points that we visit
          for each step.
    xydir - an array of directions. Wrapping the square can be viewed
            as a translation, each time we translate we record
            the opposite of the translation directions.
    entered - True if we did not exceed maxIt.
    '''

    xydir = np.zeros(n, dtype=object)
    XOut = np.zeros((n+1,4))
    XIn = np.zeros((n,4))
    XOut[0] = X
    entered = 1

    for m in range(n):
        XIn[m], xydir[m], entered = SOutToSIn(XOut[m], R, maxIt,tol)
        if 1 - entered:
            break
        XOut[m+1] = SInToSOut(XIn[m], b)

    return XOut[:m+1], XIn[:m], xydir[:m], entered


def plotCircles(ax, R, xydir):
    '''
    Helper function that draws circles on a given plt axis.

    Input:
    ax - the matplotlib axes object we want to draw to
    R - the radius of the circles
    xydir - an array of directions that is output from 
            the function orbit

    Output:
    out - the axis ax with the circles plotted
    '''

    from matplotlib.patches import Circle

    shifts = np.array([ np.sum(xy,axis=0) for xy in xydir])
    shifts = np.add.accumulate(shifts,0)
    shifts = np.unique(shifts, axis=0)
 
    for a,b in shifts:
        ax.add_patch(Circle((a+1/2,b+1/2), R, color="black", linestyle=":", fill=False))

    out = ax.add_patch(Circle((1/2,1/2), R, color="black", linestyle=":", fill=False))

    return out


def plotTrajectory(ax, XOut, XIn, xydir, **kwargs):
    '''
    Helper function to plot trajectories from the orbit function.

    Input:
    ax - the plt axis to plot to
    XOut, XIn, xydir - see the orbit function
    kwargs - to pass to the plotting function

    OUtput:
    out - the axis ax with the plotted trajectory
    '''

    XO = XOut.copy()
    XI = XIn.copy()

    shifts = np.array([ np.sum(xy,axis=0) for xy in xydir])
    shifts = np.add.accumulate(shifts,0)
    
    XI[:,:2] += shifts
    XO[:,:2] += np.vstack([np.zeros(2), shifts])

    n = XI.shape[0] + XO.shape[0]
    Xs = np.zeros((n,4))
    Xs[::2] = XO
    Xs[1::2] = XI

    out = ax.plot(Xs[:,0], Xs[:,1], "b", **kwargs)

    
    return out


def prettyPlotTrajectory(ax, XOut, XIn, xydir):
    '''
    Helper function to plot trajectories from the orbit function,
    except linear motion is blue and motion in a magnetic field
    is in red.

    Input:
    ax - the plt axis to plot to
    XOut, XIn, xydir - see the orbit function
    kwargs - to pass to the plotting function

    OUtput:
    out - the axis ax with the plotted trajectory
    '''

    XO = XOut.copy()
    XI = XIn.copy()

    shifts = np.array([ np.sum(xy,axis=0) for xy in xydir])
    shifts = np.add.accumulate(shifts,0)
    
    XI[:,:2] += shifts
    XO[:,:2] += np.vstack([np.zeros(2), shifts])

    n = XI.shape[0] + XO.shape[0]
    Xs = np.zeros((n,4))
    Xs[::2] = XO
    Xs[1::2] = XI

    for k in range(Xs.shape[0]//2):
        ax.plot(Xs[2*k:2*k+2,0], Xs[2*k:2*k+2,1], "b")
        ax.plot(Xs[2*k+1:2*k+3,0], Xs[2*k+1:2*k+3,1], "r")
    
    out = ax.plot(Xs[0,0],Xs[0,1],"ks")
    
    return out


def preparePoincare(ax, S='out', **kwargs):
    '''
    Helper function to draw the Poincare section
    of the system.

    Input:
    ax - the plt axis to plot to
    S - which Poincare section to draw, by default
        it is the outward pointing one.
    kwargs to pass to the xticks and yticks

    Output:
    out - the axis with the Poincare section drawn 
    '''

    from matplotlib.patches import Rectangle

    H = np.pi/2 * np.sqrt(2)

    if S == 'out':
        artists = [
            Rectangle((-np.pi,   np.pi/2),   H, H, angle=45, alpha=0.5),
            Rectangle((-np.pi,-3*np.pi/2), 5*H, H, angle=45, alpha=0.5),
            Rectangle(( np.pi,-3*np.pi/2),   H, H, angle=45, alpha=0.5)
            ]
    elif S == 'in':
        artists = [
            Rectangle((-np.pi,  -np.pi/2), 3*H, H, angle=45, alpha=0.5),
            Rectangle((     0,-3*np.pi/2), 3*H, H, angle=45, alpha=0.5)
            ]

    for i in artists:
        ax.add_artist(i)

    ticks = np.linspace(-1,1,5)*np.pi
    labels = ["$-\pi$", "$-\\frac{\pi}{2}$" , "$0$", "$\\frac{\pi}{2}$" , "$\pi$"]
    ax.set_xticks(ticks, labels, **kwargs)
    ax.set_yticks(ticks, labels, **kwargs)
    ax.grid(color='k', linestyle=':')

    ax.set_xlim([-np.pi,np.pi])
    out = ax.set_ylim([-np.pi,np.pi])

    return out


def plotPoincareOut(ax, XOut, **kwargs):
    '''
    Helper function to plot points on the Poincare
    section.

    Input:
    ax - the plt axis to plot to
    XOut - see the orbit function
    kwargs - to pass to the preparePoincare function

    Output:
    out - the axis with the poincare section and points
    '''

    preparePoincare(ax, S='out', **kwargs)

    thetas = np.arctan2(XOut[:,1]-1/2,XOut[:,0]-1/2)
    phis = np.arctan2(XOut[:,3],XOut[:,2])
    out = ax.plot(thetas, phis,"rs")

    return out


def circleFit(X):
    '''
    Function to fit an n-dimensional circle to data points
    via the method of I. Coope (1993)
    
    Input:
    X - (n,m) array, n is the dimension of the circle,
        m is the number of data points

    Output:
    c - the center of the circle
    r - the radius
    '''

    B = np.pad(X, ((0, 1), (0, 0)), constant_values=(1,)).T
    d = np.linalg.norm(X, axis=0)**2

    Y = np.linalg.lstsq(B, d, rcond=-1)[0]

    c = Y[:-1]/2
    r = np.sqrt(Y[-1] + np.linalg.norm(c)**2)

    return c, r


def LZ76(ss):
    """
    Simple script implementing Kaspar & Schuster's algorithm for
    Lempel-Ziv complexity (1976 version).
    
    If you use this script, please cite the following paper containing a sample
    use case and further description of the use of LZ in neuroscience:
    
    Dolan D. et al (2018). The Improvisational State of Mind: A Multidisciplinary
    Study of an Improvisatory Approach to Classical Music Repertoire Performance.
    Front. Psychol. 9:1341. doi: 10.3389/fpsyg.2018.01341
    
    Pedro Mediano and Fernando Rosas, 2019

    Calculate Lempel-Ziv's algorithmic complexity using the LZ76 algorithm
    and the sliding-window implementation.

    Reference:

    F. Kaspar, H. G. Schuster, "Easily-calculable measure for the
    complexity of spatiotemporal patterns", Physical Review A, Volume 36,
    Number 2 (1987).

    Comment from Albert: we removed the .flatten() for ss because it was easier to work with

    Input:
      ss -- array of integers

    Output:
      c  -- integer
    """

    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    n = len(ss)
    while True:
        if ss[i + k - 1] == ss[l + k - 1]:
            k = k + 1
            if l + k > n:
                c = c + 1
                break
        else:
            if k > k_max:
               k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c

