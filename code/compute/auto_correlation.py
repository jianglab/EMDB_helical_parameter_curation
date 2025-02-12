import numpy as np
import math, random
from scipy.ndimage import map_coordinates
from scipy.signal import correlate2d
from scipy.optimize import minimize
from scipy.fft import rfft2, irfft2, fftshift
from scipy.spatial import cKDTree as KDTree

from trackpy import locate, refine_com


#########################
# Cylindrical Projection
#########################
def normalize(data, percentile=(0, 100)):
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    data2 = (data-vmin)/(vmax-vmin)
    return data2


def cylindrical_projection(map3d, da=1, dz=1, dr=1, rmin=0, rmax=-1, interpolation_order=1):
    # da: degree
    # dr/dz/rmin/rmax: pixel
    assert(map3d.shape[0]>1)
    nz, ny, nx = map3d.shape
    if rmax<=rmin:
        rmax = min(nx//2, ny//2)
    assert(rmin<rmax)
    
    theta = (np.arange(0, 360, da, dtype=np.float32) - 90) * np.pi/180.
    #z = np.arange(0, nz, dz)    # use entire length
    n_theta = len(theta)
    z = np.arange(max(0, nz//2-n_theta//2*dz), min(nz, nz//2+n_theta//2*dz), dz, dtype=np.float32)    # use only the central segment 

    cylindrical_proj = np.zeros((len(z), len(theta)), dtype=np.float32)
    for r in np.arange(rmin, rmax, dr, dtype=np.float32):
        z_grid, theta_grid = np.meshgrid(z, theta, indexing='ij', copy=False)
        y_grid = ny//2 + r * np.sin(theta_grid)
        x_grid = nx//2 + r * np.cos(theta_grid)
        coords = np.vstack((z_grid.flatten(), y_grid.flatten(), x_grid.flatten()))
        cylindrical_image = map_coordinates(map3d, coords, order=interpolation_order, mode='nearest').reshape(z_grid.shape)
        cylindrical_proj += cylindrical_image * r
    cylindrical_proj = normalize(cylindrical_proj)

    return cylindrical_proj

def make_square_shape(cylproj):
    nz, na = cylproj.shape
    if nz<na:
        zeros_top = np.zeros((na//2-nz//2, na))
        zeros_bottom = np.zeros((na-nz-zeros_top.shape[0], na))
        ret = cylproj-cylproj[[0,-1], :].mean()  # subtract the mean values of top/bottom rows
        ret = np.vstack((zeros_top, ret, zeros_bottom))
    elif nz>na:
        row0 = nz//2-na//2
        ret = cylproj[row0:row0+na]
    else:
        ret = cylproj
    return ret

#########################
# Auto‑correlation
#########################

def auto_correlation(data, sqrt=False, high_pass_fraction=0):
    from scipy.signal import correlate2d
    fft = np.fft.rfft2(data)
    product = fft*np.conj(fft)
    if sqrt: product = np.sqrt(product)
    if 0<high_pass_fraction<=1:
        nz, na = product.shape
        Z, A = np.meshgrid(np.arange(-nz//2, nz//2, dtype=float), np.arange(-na//2, na//2, dtype=float), indexing='ij')
        Z /= nz//2
        A /= na//2
        f2 = np.log(2)/(high_pass_fraction**2)
        filter = 1.0 - np.exp(- f2 * Z**2) # Z-direction only
        product *= np.fft.fftshift(filter)
    corr = np.fft.fftshift(np.fft.irfft2(product))
    corr -= np.median(corr, axis=1, keepdims=True)
    corr = normalize(corr)
    if sqrt:
        corr = np.power(np.log1p(corr), 1/3)   # make weaker peaks brighter
        corr = normalize(corr)
    return corr

#########################
# Peak Detection in ACF
#########################

def find_peaks(acf, da, dz, peak_width=9.0, peak_height=9.0, minmass=1.0, max_peaks=71):
    from trackpy import locate, refine_com
    # diameter: fraction of the maximal dimension of the image (acf)
    diameter_height = int(peak_height/dz+0.5)//2*2+1
    diameter_width = int(peak_width/da+0.5)//2*2+1
    pad_width = diameter_width * 3
    acf2 = np.hstack((acf[:, -pad_width:], acf, acf[:, :pad_width]))   # to handle peaks at left/right edges
    
    # try a few different shapes around the starting height/width
    params = []
    for hf, wf in ((1, 1), (1, 2), (0.5, 0.5), (0.5, 1)):
        params += [(int(diameter_height*hf+0.5)//2*2+1, int(diameter_width*wf+0.5)//2*2+1)]
    while True:
        results = []
        for h, w in params:
            if h<1 or w<1: continue
            try:
                f = locate(acf2, diameter=(h, w), minmass=minmass, separation=(h*2, w*2))
                if len(f):
                    results.append((f["mass"].sum()*np.power(len(f), -0.5), len(f), f, h, w))
                    try:
                        f_refined = refine_com(raw_image=acf2, image=acf2, radius=(h//2, w//2), coords=f)    # radius must be even integers?
                        results.append((f_refined["mass"].sum()*np.power(len(f_refined), -0.5), len(f_refined), f_refined, h, w))
                    except:
                        pass
            except:
                pass
            if len(results) and results[-1][1] > 31: break
        results.sort(key=lambda x: x[0], reverse=True)
        if len(results) and results[0][1]>3: break
        minmass *= 0.9
        if minmass<0.1: return None, None
    f = results[0][2]

    f.loc[:, 'x'] -= pad_width
    f = f.loc[ (f['x'] >= 0) & (f['x'] < acf.shape[1]) ]
    f = f.sort_values(["mass"], ascending=False)[:max_peaks]
    peaks = np.zeros((len(f), 2), dtype=float)
    peaks[:, 0] = f['x'].values - acf.shape[1]//2    # pixel
    peaks[:, 1] = f['y'].values - acf.shape[0]//2    # pixel
    peaks[:, 0] *= da  # the values are now in degree
    peaks[:, 1] *= dz  # the values are now in Angstrom
    return peaks, f["mass"]

#########################
# Lattice Fitting getHelicalLattice and getGenericLattice
#########################

def getHelicalLattice(peaks):
    if len(peaks) < 3:
        #st.warning(f"only {len(peaks)} peaks were found. At least 3 peaks are required")
        return (0, 0, 1)

    x = peaks[:, 0]
    y = peaks[:, 1]

    ys = np.sort(y)
    vys = ys[1:] - ys[:-1]
    vy = np.median(vys[np.abs(vys) > 1e-1])
    j = np.around(y / vy)
    nonzero = j != 0
    if np.count_nonzero(nonzero)>0:
        rise = np.median(y[nonzero] / j[nonzero])
        if np.isnan(rise):
            #st.warning(f"failed to detect rise parameter. all {len(peaks)} peaks are in the same row?")
            return (0, 0, 1)
    else:
        #st.warning(f"failed to detect rise parameter. all {len(peaks)} peaks are on the equator?")
        return (0, 0, 1)

    cn = 1
    js = np.rint(y / rise)
    spacing = []
    for j in sorted(list(set(js))):
        x_j = x[js == j]
        if len(x_j) > 1:
            x_j.sort()
            spacing += list(x_j[1:] - x_j[:-1])
    if len(spacing):
        best_spacing = max(0.01, np.median(spacing)) # avoid corner case crash
        cn_f = 360. / best_spacing
        expected_spacing = 360./round(cn_f)
        if abs(best_spacing - expected_spacing)/expected_spacing < 0.05:
            cn = int(round(cn_f))

    js = np.rint(y / rise)
    above_equator = js > 0
    if np.count_nonzero(above_equator)>0:
        min_j = js[above_equator].min()  # result might not be right if min_j>1
        vx = sorted(x[js == min_j] / min_j, key=lambda x: abs(x))[0]
        x2 = np.reshape(x, (len(x), 1))
        xdiffs = x2 - x2.T
        y2 = np.reshape(y, (len(y), 1))
        ydiffs = y2 - y2.T
        selected = (np.rint(ydiffs / rise) == min_j) & (np.rint(xdiffs / vx) == min_j)
        best_vx = np.mean(xdiffs[selected])
        if best_vx > 180: best_vx -= 360
        best_vx /= min_j
        twist = best_vx
        if cn>1 and abs(twist)>180./cn:
            if twist<0: twist+=360./cn
            elif twist>0: twist-=360./cn
        if np.isnan(twist):
            #st.warning(f"failed to detect twist parameter using {len(peaks)} peaks")
            return (0, 0, 1)
    else:
        #st.warning(f"failed to detect twist parameter using {len(peaks)} peaks")
        return (0, 0, 1)

    return (twist, rise, int(cn))

def getGenericLattice(peaks):
    if len(peaks) < 3:
        #st.warning(f"only {len(peaks)} peaks were found. At least 3 peaks are required")
        return (0, 0, 1)

    from scipy.spatial import cKDTree as KDTree

    mindist = 10 # minimal inter-subunit distance
    minang = 15 # minimal angle between a and b vectors
    epsilon = 0.5

    def angle(v1, v2=None):  # angle between two vectors, ignoring vector polarity
        p = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        p = np.clip(abs(p), 0, 1)
        ret = np.rad2deg(np.arccos(p))  # 0<=angle<90
        return ret

    def distance(v1, v2):
        d = math.hypot(v1[0] - v2[0], v1[1] - v2[1])
        return d

    def onEquator(v, epsilon=0.5):
        # test if b vector is on the equator
        if abs(v[1]) > epsilon: return 0
        return 1
    
    def set_to_periodic_range(v, min=-180, max=180):
        from math import fmod
        tmp = fmod(v-min, max-min)
        if tmp>=0: tmp+=min
        else: tmp+=max
        return tmp

    def pickTriplet(kdtree, index=-1):
        '''
        pick a point as the origin and find two points closest to the origin
        '''
        m, n = kdtree.data.shape  # number of data points, dimension of each data point
        if index < 0:
            index = random.randint(0, m - 1)
        origin = kdtree.data[index]
        distances, indices = kdtree.query(origin, k=m)
        first = None
        for i in range(1, m):
            v = kdtree.data[indices[i]]
            #if onEquator(v - origin, epsilon=epsilon):
            #    continue
            first = v
            break
        second = None
        for j in range(i + 1, m):
            v = kdtree.data[indices[j]]
            #if onEquator(v - origin, epsilon=epsilon):
            #    continue
            ang = angle(first - origin, v - origin)
            dist = distance(first - origin, v - origin)
            if dist > mindist and ang > minang:
                second = v
                break
        return (origin, first, second)

    def peaks2NaNbVaVbOrigin(peaks, va, vb, origin):
        # first find indices of each peak using current unit cell vectors
        A = np.vstack((va, vb)).transpose()
        b = (peaks - origin).transpose()
        x = np.linalg.solve(A, b)
        NaNb = np.around(x)
        # then refine unit cell vectors using these indices
        good = np.abs(x-NaNb).max(axis=0) < 0.1 # ignore bad peaks
        one = np.ones((1, (NaNb[:, good].shape)[1]))
        A = np.vstack((NaNb[:, good], one)).transpose()
        (p, residues, rank, s) = np.linalg.lstsq(A, peaks[good, :], rcond=-1)
        va = p[0]
        vb = p[1]
        origin = p[2]
        err = np.sqrt(sum(residues)) / len(peaks)
        return {"NaNb": NaNb, "a": va, "b": vb, "origin": origin, "err": err}

    kdt = KDTree(peaks)

    bestLattice = None
    minError = 1e30
    for i in range(len(peaks)):
        origin, first, second = pickTriplet(kdt, index=i)
        if first is None or second is None: continue
        va = first - origin
        vb = second - origin

        lattice = peaks2NaNbVaVbOrigin(peaks, va, vb, origin)
        lattice = peaks2NaNbVaVbOrigin(peaks, lattice["a"], lattice["b"], lattice["origin"])
        err = lattice["err"]
        if err < minError:
            dist = distance(lattice['a'], lattice['b'])
            ang = angle(lattice['a'], lattice['b'])
            if dist > mindist and ang > minang:
                minError = err
                bestLattice = lattice

    if bestLattice is None:
        # assume all peaks are along the same line of an arbitrary direction
        # fit a line through the peaks
        from scipy.odr import Data, ODR, unilinear
        x = peaks[:, 0]
        y = peaks[:, 1]
        odr_data = Data(x, y)
        odr_obj = ODR(odr_data, unilinear)
        output = odr_obj.run()
        x2 = x + output.delta   # positions on the fitted line
        y2 = y + output.eps
        v0 = np.array([x2[-1]-x2[0], y2[-1]-y2[0]])
        v0 = v0/np.linalg.norm(v0, ord=2)   # unit vector along the fitted line
        ref_i = 0
        t = (x2-x2[ref_i])*v0[0] + (y2-y2[ref_i])*v0[1] # coordinate along the fitted line
        t.sort()
        spacings = t[1:]-t[:-1]
        a = np.median(spacings[np.abs(spacings)>1e-1])
        a = v0 * a
        if a[1]<0: a *= -1
        bestLattice = {"a": a, "b": a}

    a, b = bestLattice["a"], bestLattice["b"]

    minLength = max(1.0, min(np.linalg.norm(a), np.linalg.norm(b)) * 0.9)
    vs_on_equator = []
    vs_off_equator = []
    maxI = 10
    for i in range(-maxI, maxI + 1):
        for j in range(-maxI, maxI + 1):
            if i or j:
                v = i * a + j * b
                v[0] = set_to_periodic_range(v[0], min=-180, max=180)
                if np.linalg.norm(v) > minLength:
                    if v[1]<0: v *= -1
                    if onEquator(v, epsilon=epsilon):
                        vs_on_equator.append(v)
                    else:
                        vs_off_equator.append(v)
    twist, rise, cn = 0, 0, 1
    if vs_on_equator:
        vs_on_equator.sort(key=lambda v: abs(v[0]))
        best_spacing = abs(vs_on_equator[0][0])
        cn_f = 360. / best_spacing
        expected_spacing = 360./round(cn_f)
        if abs(best_spacing - expected_spacing)/expected_spacing < 0.05:
            cn = int(round(cn_f))
    if vs_off_equator:
        vs_off_equator.sort(key=lambda v: (abs(round(v[1]/epsilon)), abs(v[0])))
        twist, rise = vs_off_equator[0]
        if cn>1 and abs(twist)>180./cn:
            if twist<0: twist+=360./cn
            elif twist>0: twist-=360./cn
    return twist, rise, int(cn)

def fitHelicalLattice(peaks, acf, da=1.0, dz=1.0):
    if len(peaks) < 3:
        #st.warning(f"WARNING: only {len(peaks)} peaks were found. At least 3 peaks are required")
        return (None, None, peaks)

    trc1s = []
    trc2s = []
    consistent_solution_found = False
    nmax = len(peaks) if len(peaks)%2 else len(peaks)-1
    for n in range(nmax, min(7, nmax)-1, -2):
        trc1 = getHelicalLattice(peaks[:n])
        trc2 = getGenericLattice(peaks[:n])
        trc1s.append(trc1)
        trc2s.append(trc2)
        if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0):
            consistent_solution_found = True
            break
    
    if not consistent_solution_found: 
        for _ in range(100):
            from random import randint, sample
            if len(peaks)//2 > 5:   # stronger peaks
                n = randint(5, len(peaks)//2)
                random_choices = sorted(sample(range(2*n), k=n))
            else:
                n = randint(3, len(peaks))
                random_choices = sorted(sample(range(len(peaks)), k=n))
            if 0 not in random_choices: random_choices = [0] + random_choices
            peaks_random = peaks[random_choices]
            trc1 = getHelicalLattice(peaks_random)
            trc2 = getGenericLattice(peaks_random)
            trc1s.append(trc1)
            trc2s.append(trc2)
            if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1):
                consistent_solution_found = True
                break
    
    if not consistent_solution_found: 
        trc_mean = consistent_twist_rise_cn_sets(trc1s, trc2s, epsilon=1)
        if trc_mean:
            _, trc1, trc2 = trc_mean
        else:
            trc1s = np.array(trc1s)
            trc2s = np.array(trc2s)
            trc1 = list(geometric_median(X=trc1s[:,:2])) + [int(np.median(trc1s[:,2]))]
            trc2 = list(geometric_median(X=trc2s[:,:2])) + [int(np.median(trc2s[:,2]))]

    twist1, rise1, cn1 = trc1
    twist1, rise1 = refine_twist_rise(acf_image=(acf, da, dz), twist=twist1, rise=rise1, cn=cn1)
    twist2, rise2, cn2 = trc2
    twist2, rise2 = refine_twist_rise(acf_image=(acf, da, dz), twist=twist2, rise=rise2, cn=cn2)
    
    return (twist1, rise1, cn1), (twist2, rise2, cn2)

def consistent_twist_rise_cn_sets(twist_rise_cn_set_1, twist_rise_cn_set_2, epsilon=1.0):
    def angle_difference(angle1, angle2):
        err = abs((angle1 - angle2) % 360)
        if err > 180: err -= 360
        err = abs(err)
        return err

    def angle_mean(angle1, angle2):
        angles = np.deg2rad([angle1, angle2])
        ret = np.rad2deg(np.arctan2( np.sin(angles).sum(), np.cos(angles).sum()))
        return ret

    def consistent_twist_rise_cn_pair(twist_rise_cn_1, twist_rise_cn_2, epsilon=1.0):
        def good_twist_rise_cn(twist, rise, cn, epsilon=0.1):
            if abs(twist)>epsilon:
                if abs(rise)>epsilon: return True
                elif abs(rise*360./twist/cn)>epsilon: return True # pitch>epsilon
                else: return False
            else:
                if abs(rise)>epsilon: return True
                else: return False
        if twist_rise_cn_1 is None or twist_rise_cn_2 is None:
            return None
        twist1, rise1, cn1 = twist_rise_cn_1
        twist2, rise2, cn2 = twist_rise_cn_2
        if not good_twist_rise_cn(twist1, rise1, cn1, epsilon=0.1): return None
        if not good_twist_rise_cn(twist2, rise2, cn2, epsilon=0.1): return None
        if cn1==cn2 and abs(rise2-rise1)<epsilon and angle_difference(twist1, twist2)<epsilon:
            cn = cn1
            rise_tmp = (rise1+rise2)/2
            twist_tmp = angle_mean(twist1, twist2)
            return twist_tmp, rise_tmp, int(cn)
        else:
            return None
    for twist_rise_cn_1 in twist_rise_cn_set_1:
        for twist_rise_cn_2 in twist_rise_cn_set_2:
            trc = consistent_twist_rise_cn_pair(twist_rise_cn_1, twist_rise_cn_2, epsilon=epsilon)
            if trc: return (trc, twist_rise_cn_1, twist_rise_cn_2)
    return None

def set_to_periodic_range(v, min=-180, max=180):
        from math import fmod
        tmp = fmod(v-min, max-min)
        if tmp>=0: tmp+=min
        else: tmp+=max
        return tmp

def refine_twist_rise(acf_image, twist, rise, cn):
    from scipy.optimize import minimize
    if rise<=0: return twist, rise
    cn = int(cn)

    acf_image, da, dz = acf_image
    
    ny, nx = acf_image.shape
    try:
        npeak = max(3, min(100, int(ny/2/abs(rise)/2)))
    except:
        npeak = 3
    i = np.repeat(range(1, npeak), cn)
    w = np.power(i, 1./2)
    x_sym = np.tile(range(cn), npeak-1) * 360./cn    
    def score(x):
        twist, rise = x
        px = np.fmod(nx//2 + i * twist/da + x_sym + npeak*nx, nx)
        py = ny//2 + i * rise/dz
        v = map_coordinates(acf_image, (py, px))
        score = -np.sum(v*w)
        return score    
    res = minimize(score, (twist, rise), method='nelder-mead', options={'xatol': 1e-4, 'adaptive': True})
    twist_opt, rise_opt = res.x
    twist_opt = set_to_periodic_range(twist_opt, min=-180, max=180)
    return twist_opt, rise_opt


# https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
def geometric_median(X, eps=1e-5):
    import numpy as np
    from scipy.spatial.distance import cdist, euclidean
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

#########################
# Main Computation Function
#########################

def compute_helical_parameters(map3d, apix, da=1.0, dz=1.0,
                               peak_width_base=9.0, peak_height_base=9.0, minmass=1.0,
                               rmin=0, rmax=-1):
    """
    Given a 3D density map (as a numpy array) and its pixel size,
    compute the helical parameters: twist (in degrees per subunit),
    rise (in Å per subunit), and cyclic symmetry (csym).
    
    Parameters:
      map3d      : 3D numpy array of the density map.
      apix : pixel size in Å/voxel.
      da         : angular step size for cylindrical projection (degrees).
      dz         : axial step size (in Å; note: converted to voxels internally).
      peak_width, peak_height: expected dimensions of peaks in the ACF.
      minmass    : minimum threshold for peak detection.
      rmin, rmax : radial range (in pixels) for the projection.
    
    Returns:
      twist, rise, csym
    """
    
    # 1. Compute cylindrical projection of the 3D map.
    cylproj = cylindrical_projection(map3d, da=da, dz=dz/apix, rmin=rmin, rmax=rmax)
    cylproj_square = make_square_shape(cylproj)
    
    # 2. Compute the auto-correlation of the cylindrical projection.
    acf = auto_correlation(cylproj_square, sqrt=False, high_pass_fraction=1.0 / cylproj_square.shape[0])
    
    # 3. Detect peaks in the auto-correlation image.

    peak_width = peak_width_base*da
    peak_height = peak_height_base*dz

    peaks, masses = find_peaks(acf, da=da, dz=dz, peak_width=peak_width, peak_height=peak_height, minmass=1)
    if peaks is None or len(peaks) < 3:
        raise ValueError("Not enough peaks were detected in the auto-correlation image.")
    

     
    npeaks_all = len(peaks)
    try:
        from kneebow.rotor import Rotor
        rotor = Rotor()
        rotor.fit_rotate(np.vstack((np.arange(len(masses)-3), masses.iloc[3:])).T )
        npeaks_guess = min(npeaks_all, rotor.get_elbow_index()+3)
    except:
        npeaks_guess = npeaks_all
    
    npeaks = npeaks_guess

    # 4. Fit the helical lattice using the detected peaks.
    trc1, trc2 = fitHelicalLattice(peaks[:npeaks], acf, da, dz)
    
    # 5. Try to combine the two candidate solutions.
    trc_mean = consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0)
    success = True if trc_mean else False

    if success:
        twist_tmp, rise_tmp, cn = trc_mean[0]
        twist_auto, rise_auto = refine_twist_rise(acf_image=(acf, da, dz), twist=twist_tmp, rise=rise_tmp, cn=cn)
        csym_auto = cn
    else:
        twist_auto, rise_auto, csym_auto = trc1
    return twist_auto, rise_auto, csym_auto