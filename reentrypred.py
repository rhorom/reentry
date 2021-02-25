#basic packages
import numpy as np
import pandas as pd

#for prediction
from skyfield.api import Topos, load, EarthSatellite
from sgp4.api import Satrec, WGS72
from scipy import integrate as ig
from scipy.special import iv
from pyatmos import download_sw, read_sw, nrlmsise00

#for Visualization
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

#Constants
mu = 3.985880e14
re = 6.378135e6
tpi = 2*np.pi
tsc = load.timescale()

#Load space weather parameters
try:
    with open('data/SW-All.txt') as a:
        print('Space weather data available')
except:
    a = download_sw(direc='data/')

swdata = read_sw('data/SW-All.txt')
a = swdata[0,0:3].astype(int)
b = swdata[-1,0:3].astype(int)
swfirs = tsc.utc(b[0], b[1], b[2], 0, 0)
swlast = tsc.utc(a[0], a[1], a[2]+1, 0, 0)
print(swfirs.utc_iso()[0:10], 'to', swlast.utc_iso()[0:10])

def period(sma, unit='sec'):
    '''
    Function to calculate the orbital period of geocentric
    object with certain semimajor axis
    
    Input:
    sma : semimajor axis [m]
    unit: unit of the calculated period. It can be 'sec',
          'min', 'hrs', or 'day'. The default is 'sec'
          
    Output:
    period: orbital period in preferred unit
    '''

    a = {'sec':1, 'min':60, 'hrs':3600, 'day':86400}
    p = tpi*np.sqrt(sma**3/mu)
    
    return p/a[unit]

def n2sma(n0, unit='revday2km', alti=True):
    '''
    Function to calculate mean altitude based on the
    mean motion
    
    Input:
    n0  : mean motion which can either be rev/day or
          rad/min. The unit is governed by 'unit' argument
          The default unit is rev/day
    unit: unit of input and output which can either be
          'revday2km', 'radmin2km', 'revday2m', and 'radmin2m'
    alti: boolean to calculate mean altitude. If False, then
          the semimajor axis is produced instead of mean altitude.
          The default is True. The default unit is km
          
    Output:
    altitude or semimajor axis in km or m
    '''

    a = {'revday2km': 42240.656,
         'revday2m' : 42240656.,
         'radmin2km': 1127.913,
         'radmin2m' : 1127913.}
    sma = a[unit]*n0**(-2/3)
    if alti:
        b = {'revday2km': re*1e-3,
             'revday2m' : re,
             'radmin2km': re*1e-3,
             'radmin2m' : re}
        return sma - b[unit]
    else:
        return sma
    
def sma2n(sma, unit='m2radmin'):
    '''
    Function to calculate mean motion based on
    semimajor axis
    
    Input:
    sma  : semimajor axis. The unit is governed by
        unit argument. By default, it is in meter
    unit : unit of conversion. It can either be
        'm2radmin', 'm2revday', 'm2radsec',
        'km2radmin','km2revday'
    
    Output:
    n0 : mean motion. The unit is governed by unit
        argument. By default, it is in rad/min
    '''
    
    a = {'m2radmin' : 1.1978801e9,
         'm2revday' : 2.7453390e11,
         'm2radsec' : 1.9964669e7,
         'km2radmin': 3.7880295e4,
         'km2revday': 8.6815243e6}
    n0 = a[unit]*sma**(-1.5)
    
    return n0

def loadtle(fnam, save=''):
    '''
    Read and extract TLEs from historical TLEs
    as DataFrame
    
    Input:
    fnam -- path of the file containing the historical TLEs
    save -- name of the output CSV file
    
    Output:
    DataFrame containing 'satnum' [NORAD ID], 'epoch' [JD],
        'ndot' [rev/day2], 'bstr' [m2/kg Rearth], 'inclo' [deg],
        'nodeo' [deg], 'ecco', 'argpo' [deg],
        'no_kozai' [rad/min], 'mean_alti [km]
    '''
    
    try:
        g = open(fnam, 'r')
    except:
        print('Input file not found')
        return
    
    out = ([],[],[],[],[],[],[],[],[])
    for r in g:
        if r[0] == '1':
            a = int(r[18:20])
            yy = 1900 + a
            if a < 50: yy += 100
            t = tsc.utc(yy, 1, 0 + float(r[20:32]))
            out[0].append(t.tt)
            out[1].append(float(r[33:43]))
            out[2].append(float(r[53]+'.'+r[54:59]+'E'+r[59:61]))
        if r[0] == '2':
            out[3].append(r[2:7])
            out[4].append(float(r[8:16]))
            out[5].append(float(r[17:25]))
            out[6].append(float('.'+r[26:33]))
            out[7].append(float(r[34:42]))
            out[8].append(float(r[52:63]))
    g.close()

    res = pd.DataFrame({'satnum'   : out[3], 
                        'epoch'    : out[0], #Julian Day
                        'date'     : tsc.tt_jd(out[0]).utc_iso(),
                        'ndot'     : out[1], #rev/day2
                        'bstr'     : out[2], #m2/kg Rearth
                        'inclo'    : out[4], #degrees
                        'nodeo'    : out[5], #degrees
                        'ecco'     : out[6], 
                        'argpo'    : out[7], #degrees
                        'no_kozai' : tpi*np.array(out[8])/1440, #rad/min
                        'mean_alti': n2sma(np.array(out[8]))
                       })
    
    res = res.sort_values(by=['epoch'])
    
    if save != '':
        res.to_csv('data/{:}.csv'.format(save), index=False)

    return(res)

def loadcsv(fnam, yy0, percent=50):
    '''
    Function to load historical TLE previously saved
    in CSV format
    
    Input:
    fnam   : path of the CSV file containing the historical TLEs
    percent: selected percentile of drag term B*
    
    Output:
    obs_data: DataFrame containing 'satnum' [NORAD ID], 'epoch' [JD],
        'ndot' [rev/day^2], 'bstr' [m^2/kg Rearth], 'inclo' [deg],
        'nodeo' [deg], 'ecco', 'argpo' [deg], 'no_kozai' [rad/min],
        'mean_alti [km], 'days' [since first epoch]
    params  : initial parameters containing 'epoch', 'date',
        'mean_alti', 'ecco', 'inclo', 'argpo', 'bstr'
    '''
    
    try:
        obs_data = pd.read_csv(fnam)
    except:
        print('Input file not found')
        return

    #yy = np.array([int(a[0:4]) for a in obs_data.date.values])
    #obs_data = obs_data[yy >= yy0]
    
    obs_data['days'] = obs_data.epoch - obs_data.epoch[2]
    dtm0 = obs_data.date.values[2][0:16]
    params = pd.DataFrame({'epoch': [obs_data.epoch[2]],
                           'date' : [dtm0],
                           'mean_alti': [np.median(obs_data.mean_alti.values[0:5])],
                           'sma'  : [n2sma(np.median(obs_data.no_kozai.values[0:5]),
                                     unit='radmin2m', alti=False)],
                           'ecco' : [np.median(obs_data.ecco.values[0:5])],
                           'inclo': [np.median(obs_data.inclo.values[0:5])],
                           'argpo': [np.median(obs_data.argpo.values[0:5])],
                           'bstr' : [np.percentile(obs_data.bstr.values, percent)]
                          })
    
    return obs_data, params

def atm_dens(t, h, hscale=False):
    '''
    Function to calculate total atmospheric density at certain
    time and height based on NRLMSISE-00. The atmospheric model
    is multidimensional and depends on the geographic/geomagnetic
    location, this function only return the density at 0deg longitude
    and 0deg latitude.
    
    Input:
    t     : date and time in isoformat
    h     : height [m]
    hscale: boolean to calculate scale height
    
    Output:
    rho : total atmospheric density [kg/m^3]
    H   : approximated scale height [km]
    '''
    
    lon, lat, alt = 0, 0, h*1e-3 #the altitude is converted to km
    atm_in, atm_out = nrlmsise00(t, lat, lon, alt, swdata)
    rho = atm_out['Density']['RHO[kg/m^3]']
        
    if hscale:
        dh = 0.01
        atm_in, atm_out2= nrlmsise00(t, lat, lon, alt+dh, swdata)
        rho2 = atm_out2['Density']['RHO[kg/m^3]']
        H = -rho*dh/(rho2-rho)
        return rho, H
    
    else:
        return rho
    
def t_factor(a, e):
    '''
    Function for defining multiplier of time step of integration.
    Used in predict().
    
    Input:
    a : semimajor axis
    e : eccentricity
    
    Output:
    tfac: multiplier
    '''
    
    return 1

def predict(a0, e0, bst, epoch0, tfac=1, max_iter=1e5, verbose=False, save=''):
    '''
    Function to predict the orbit contraction due to atmospheric drag.
    This function is based on the semianalytical formulation by
    King-Hele (1964) summarized in the appendix of Frey et al. (2019).
    
    Input:
    a0    : initial semimajor axis [m]
    e0    : initial eccentricity
    bst   : initial drag term (B* in TLE) [m^2/kg R_earth]
    epoch0: initial epoch [JD]
    tfac  : multiplying factor of the integration timestep.
        the default value is 1*period
    max_iter : maximum iteration
    verbose  : boolean to print record per 100 iteration
    
    Output:
    rec : DataFrame containing 'epoch' [JD], 'sma' [m], 
        'ecc', 'peri' [m], 'rho' [km/m^3]
        
    References:
    
    '''
    
    a, e = a0, e0
    h_pe = a - a*e - re
    B = 12.741621*bst
    epoch = epoch0
    t_max = swlast.tt
    tx = tsc.tt_jd(epoch).utc_iso()[0:16]
    
    print('Start of Simulation')
    print('Date       : {:}'.format(tx))
    print('sma [km]   : {:.1f}'.format(1e-3*a0))
    print('ecc        : {:.4f}'.format(e0))
    print('B [m2/kg]  : {:.4f}'.format(B))
    print('h_pe [km]  : {:.1f}\n'.format(1e-3*h_pe))
    
    ca1 = np.array([[    1,     0,     0,     0,     0,     0,     0],
                    [    0,     2,     0,     0,     0,     0,     0],
                    [  .75,     0,   .75,     0,     0,     0,     0],
                    [    0,   .75,     0,   .25,     0,     0,     0],
                    [.3281,     0, .4375,     0, .1094,     0,     0],
                    [    0, .4688,     0, .2344,     0, .0469,     0]
                   ])
    ce1 = np.array([[     0,      1,      0,      0,      0,      0,      0],
                    [   .50,      0,    .50,      0,      0,      0,      0],
                    [     0,  -.625,      0,   .125,      0,      0,      0],
                    [-.3125,      0,   -.25,      0, -.0625,      0,      0],
                    [     0, -.1406,      0, -.0078,      0,  .0234,      0],
                    [-.0703,      0, -.0742,      0,  .0078,      0,  .0117]
                   ])
    ca2 = np.array([[    .5,  .0625,  .0351,  .0366,  .0561,  .1135],
                    [     0,    -.5, -.1875, -.1758, -.2563, -.5047],
                    [     0,  .1875,  .5859,  .3296,  .3653,  .5497],
                    [     0,      0,  .1875, -.5859, -.0513,  .6345],
                    [     0,      0, -.0586,-1.8237,  .6697,-1.3518],
                    [     0,      0,      0, -.1758, 6.6138,-3.9509],
                    [     0,      0,      0, 0.0513, 6.7740,-29.326],
                    [     0,      0,      0,      0, 0.2563,-51.045],
                    [     0,      0,      0,      0, -.0721,-31.141],
                    [     0,      0,      0,      0,      0, -.5047],
                    [     0,      0,      0,      0,      0,  .1388]
                   ])
    ce2 = np.array([[   0.5, -.1875, -.0586, -.0513, -.0721, -.1388],
                    [     0,   -.25,  .2812,  .1465,  .1794,  .3244],
                    [     0,  .1875,  .3047, -.1977,  .0320,  .2902],
                    [     0,      0,  .0937,-1.4648,  .1794,  .9517],
                    [     0,      0, -.0586, -.7397, 3.7715,-2.0223],
                    [     0,      0,      0, -.0879, 7.7161,-17.779],
                    [     0,      0,      0,  .0513, 2.4930,-39.045],
                    [     0,      0,      0,      0,  .1282,-45.941],
                    [     0,      0,      0,      0, -.0721,-10.903],
                    [     0,      0,      0,      0,      0, -.2523],
                    [     0,      0,      0,      0,      0,  .1388]
                   ])
    n_iter = 0
    rec = pd.DataFrame()
    t_init = tsc.now()
        
    while h_pe > 5e4:
        n_iter += 1
        #h_ap = a + a*e - re
        h_pe = a - a*e - re
        tx = tsc.tt_jd(epoch).utc_iso()[0:16]
        
        if h_pe < 0:
            break
            
        if h_pe < 1.5e5:
            tfac = min(tfac, 2)
            
        if verbose:
            if n_iter%100 == 0:
                print(n_iter, tx, '{:.1f}'.format(h_pe*0.001))

        if e <= 0.0001:
            rho = atm_dens(tx, h_pe, hscale=False)
            ka = B*rho*np.sqrt(mu*a)
            F_a = -ka
            F_e = 0
            
        elif e <= 0.2:
            rho, hsc = atm_dens(tx, h_pe, hscale=True)
            z = 1e-3*a*e/hsc
            et = e**np.arange(6)
            it = iv(range(7), z)
            
            ka = B*rho*np.sqrt(mu*a)
            ke = ka/a
            F_a = -ka*np.exp(-z)*np.matmul(np.matmul(et, ca1), it)
            F_e = -ke*np.exp(-z)*np.matmul(np.matmul(et, ce1), it)
            
        else:
            rho, hsc = atm_dens(tx, h_pe, hscale=True)
            z = 1e-3*a*e/hsc
            x = 1/(z*(1-e*e))
            rt = x**np.arange(6)
            et = e**np.arange(11)
            
            ka = B*rho*np.sqrt(mu*a)
            ke = ka/a
            
            ga = np.sqrt(0.6366198/z*(1 + e)**3/(1 - e))
            ge = ga*(1 - e)
            F_a = -ka*ga*np.matmul(rt, np.matmul(et, ca2))
            F_e = -ke*ge*np.matmul(rt, np.matmul(et, ce2))
            
        rec = rec.append(pd.DataFrame({'epoch': [epoch],
                                       'sma'  : [a],
                                       'ecc'  : [e],
                                       'peri' : [h_pe],
                                       'rho'  : [rho]
                                      }))

        if tfac == 'var':
            tfac = t_factor(a, e)
            
        P = period(a, unit='sec')
        delta_t = tfac*P
        delta_d = delta_t*1.1574074e-05
        a += F_a*delta_t
        e += F_e*delta_t
        e = max(0, e)
        epoch += delta_d
        
        if n_iter > max_iter:
            break
            
        if (epoch - t_max) > 0:
            print('Stopped. SW data unavailable')
            break

    if save != '':
        rec.to_csv('data/{:}.csv'.format(save), index=False)
        
    print('\nEnd of Simulation')
    print('Run Time   : {:.3f} sec'.format((tsc.now()-t_init)*86400))
    print('N_iter     : {:}'.format(n_iter))
    print('Reentry on : {:}'.format(tx))
    print('Speed[km/s]: {:.3f}'.format(1e-3*F_a))
    print('Lifetime[d]: {:.3f}'.format(epoch-epoch0))
    print()

    return rec

def predictTLE(sat, rec, bstar=''):
    '''
    Function to predict/extrapolate TLE given that the
    orbit experience contraction as predicted using predict()
    function.
    
    Input:
    sat  : TLE
    rec  : results from predict()
    bstar: drag term (B* in TLE) [m^2/kg R_earth]
    
    Output:
    sat2 : extrapolated TLE at reentry time
    '''
    
    t = tsc.tt_jd(rec.epoch.values)
    dt = t - sat.epoch

    a = dt>0
    sma = np.append(np.interp(0, dt, rec.sma.values), rec.sma.values[a])
    ecco = np.append(np.interp(0, dt, rec.ecc.values), rec.ecc.values[a])
    dt = dt[a]

    argpo = sat.model.argpo
    nodeo = sat.model.nodeo
    inclo = sat.model.inclo
    cosi = np.cos(inclo)
    no_kozai = sma2n(sma, unit='m2radmin')
    aepo = 1.08263e-3*(re/sma/(1-ecco*ecco))**2 #J2*(re/po)**2 from Spacetrak Report 3

    for i in range(len(dt)-1):
        f = aepo[i]*no_kozai[i]*cosi
        dtmin = 1440*(dt[i+1] - dt[i])
        nodeo += -1.5*f*dtmin
        argpo += 0.75*f*(5*cosi*cosi-1)*dtmin

    nodeo = nodeo - tpi*np.floor(nodeo/tpi)
    argpo = argpo - tpi*np.floor(argpo/tpi)
    epoch = t[-1] - tsc.utc(1949,12,31,0)
    ndot = (no_kozai[-1] - no_kozai[-2])/(dt[-1] - dt[-2])
    
    if bstar == '':
        bstar = sat.model.bstar
        
    satp = Satrec()
    satp.sgp4init(
        WGS72, 'i',
        sat.model.satnum,
        epoch, bstar, ndot, 0.0,
        ecco[-1], argpo, inclo,
        0.0, no_kozai[-1], nodeo)

    return EarthSatellite.from_satrec(satp, tsc)

def viewhistory(obse, save=''):
    '''
    Input:
    
    Output:
    
    '''
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(6,6), dpi=100, facecolor='w')
    t0 = tsc.tt_jd(int(obse.epoch.values[0])-0.5)
    t1 = obse.epoch.values - t0.tt
    
    ax1.plot(t1, obse.no_kozai.values, '.k', ms=3)
    ax1.set_ylabel('$n$')
    ax1.grid(linestyle=':', color='gray')
    
    ax2.plot(t1, obse.ecco.values, '.k', ms=3)
    ax2.set_ylabel('$e$')
    ax2.grid(linestyle=':', color='gray')
    
    ax3.plot(t1, obse.inclo.values, '.k', ms=3)
    ax3.set_ylabel('$i$')
    ax3.grid(linestyle=':', color='gray')
    
    ax4.plot(t1, 1000*obse.bstr.values, '.k', ms=3)
    ax4.set_ylabel('$10^3B*$')
    ax4.set_ylim(0,5)
    ax4.grid(linestyle=':', color='gray')
    
    plt.tight_layout()
    
    if save != '':
        plt.savefig('data/{:}.png'.format(save))
        plt.savefig('data/{:}.pdf'.format(save))
    plt.show()
    
    return
    
def viewrec(rec, obse=[], save='', peri=False):
    '''
    To display the result of orbit contraction model and
    its residual with respect to the observed parameter.
    
    Input:
    rec  : results from predict()
    obse : observed parameter from loadcsv()
    save : filename of the resulted graphics (in PNG and PDF)
    peri : toogle to show perigee height
    Output:
    
    '''
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6,6), dpi=100, facecolor='w')
    ax3 = ax2.twinx()
    
    t0 = tsc.tt_jd(rec.epoch.values[0])

    t1 = rec.epoch.values - t0.tt
    h1 = 1e-3*(rec.sma.values - re)
    
    fma = 'Start\n{:}\n$a_0$ : $R_E$ + {:.0f}\n$e_0$ : {:.3f}\n\n'
    fmb = 'End  \n{:}\n$a_f$ : $R_E$ + {:.0f}\n$p_f$ : {:.0f}\n$e_f$ : {:.3f}'
    
    txa = ''
    txb = fmb.format(tsc.tt_jd(rec.epoch.values[-1]).utc_iso()[0:16],
                     h1[-1],
                     1e-3*rec.peri.values[-1],
                     rec.ecc.values[-1])
    
    plot_obse = len(obse) > 1
    if plot_obse:
        t2 = obse.epoch.values - t0.tt
        h2 = np.interp(t2, t1, h1)
        e2 = np.interp(t2, t1, rec.ecc.values)
        ea = 100*(h2/obse.mean_alti.values - 1)
        ee = 100*(e2/obse.ecco.values - 1)

        fmc = 'Res(a): {:.1f}% to {:.1f}%\nRes(e): {:.1f}% to {:.1f}%'
        txa = fma.format(tsc.tt_jd(obse.epoch.values[0]).utc_iso()[0:16],
                         obse.mean_alti.values[0],
                         obse.ecco.values[0])
        txc = fmc.format(ea.min(), ea.max(), ee.min(), ee.max())
        ax1.plot(t2, ea, '--k', ms=3, label='altitude')
        ax1.plot(t2, ee, '--r', ms=3, label='eccentricity')
        ax1.text(0.025*t1[-1], -9, txc, va='bottom', ha='left', fontsize='small')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.text(0.05*t1[-1], 9, obse.satnum.values[0], va='top', ha='left',
                 fontweight='bold')
        ax2.plot(t2, obse.mean_alti.values, '.k', ms=3)
        ax3.plot(t2, 100*obse.ecco.values, '.r', ms=3)
        
    if peri:
        h2 = 1e-3*rec.peri.values
        h3 = 2*h1 - h2
        ax2.fill_between(t1, h3, h2, color='lightgray')
        
    ax2.plot(t1, h1, '--k', label='alti')
    ax2.set_xlabel('Days since {:}'.format(t0.utc_iso()[0:10]))
    ax2.set_ylabel('$h$ [km]')
    ax2.set_ylim(100, 1.01*h1[0])
    ax2.text(0.025*t1[-1], 105 + 0.05*h1[0], txa+txb, va='bottom', ha='left', fontsize='x-small')
    
    ax3.plot(t1, 100*rec.ecc.values, '--r', label='ecce')
    ax3.set_ylabel(r'$100\times e$', color='r')
    ax3.tick_params(axis='y', colors='r')
    
    ax1.plot(t1, t1*0, '-k')
    ax1.set_ylabel('Residual [%]')
    ax1.set_ylim(-10, 10)
    ax1.set_yticks(np.linspace(-10,10,5))
    
    plt.tight_layout()
    
    if save != '':
        plt.savefig('data/{:}.png'.format(save))
        plt.savefig('data/{:}.pdf'.format(save))

    plt.show()
    return

def viewtrack(sat, epoch, revs, output=False, add='', save=''):
    '''
    Displaying ground track of satellite at certain epoch according
    to its TLE.
    
    Input:
    sat   :
    epoch : starting epoch in Julian Day
    revs  : number of revolutions to be displayed
    output: boolean to return DataFrame output
    save  : filename of the resulted graphics (in PNG and PDF)
    
    Output:
    
    '''
    
    plt.figure(figsize=(10,5), dpi=100, facecolor='w')
    pr = ccrs.PlateCarree()
    ax = plt.axes(projection=pr)
    ax.set_extent([-180,180,-90,90], crs=pr)
    ax.coastlines('50m', linewidth=0.6)

    if type(sat) != list:
        sat = [sat]
        
    for s in sat:
        p = tpi/s.model.no_kozai
        minutes = np.arange(0, revs*p, 0.25)
        t = tsc.tt_jd(epoch + minutes/1440)
        pos = s.at(t)
        sub = pos.subpoint()
        lon = sub.longitude.degrees
        lat = sub.latitude.degrees
        elv = sub.elevation.km
        a = np.argmin(elv)
        b = np.arange(0, len(t), 20)
        
        ax.scatter(lon, lat, s=3, c=elv, cmap='Reds_r')
        ax.scatter(lon[b], lat[b], s=5, c=elv[b], cmap='Reds_r', 
                   vmin=min(elv), vmax=max(elv), marker='+')
        
        ax.plot(lon[a], lat[a], 'sr')
        tx = '{:8d}\n{:8.3f}\n{:8.3f}\n{:8.0f}'.format(s.model.satnum,
                                                  lon[a],lat[a],elv[a])
        ax.text(lon[a]-5, lat[a], tx, va='top', ha='right',
                fontsize='x-small', color='r',
                bbox=dict(fc='w', ec='w', alpha=0.75))
    
    if add != '':
        ax.plot(add[0], add[1], 'sb')
        ax.text(add[0]-5, add[1], '{:.1f}\n{:.1f}'.format(add[0], add[1]),
                fontsize='x-small', va='top', ha='right', color='b',
                bbox=dict(fc='w', ec='w', alpha=0.75))
        
    tx = '{:} UT + {:.0f} min'.format(t[0].utc_iso()[0:16], minutes[-1])
    ax.text(-170, -85, tx, ha='left', va='bottom',
            fontsize='large',
            bbox=dict(fc='w', ec='w', alpha=0.75))
    ax.gridlines(linestyle=':', color='gray')

    #plt.tight_layout()
    if save != '':
        plt.savefig('data/{:}.png'.format(save))
        plt.savefig('data/{:}.pdf'.format(save))
        
    plt.show()
    if output:
        rec = pd.DataFrame({'Time': t.utc_iso(),
                            'Lon': lon,
                            'Lat': lat,
                            'Elv': elv})
        return rec
    
    
def viewsw(epoch):
    t0 = tsc.tt_jd(int(epoch)-0.5)

    yy = swdata[:,0].astype(int)
    mm = swdata[:,1].astype(int)
    dd = swdata[:,2].astype(int)
    t1 = tsc.utc(yy, mm, dd).tt
    
    a = np.where(t1 - t0.tt >= 0)[0]

    ap = swdata[a,19].astype(int)
    f1 = swdata[a,24].astype(float)
    f2 = swdata[a,26].astype(float)

    n = len(f1)
    x = t1[a] - t0.tt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5,4), dpi=100, facecolor='w')

    ax1.plot(x, f2, '-', color='salmon', lw=0.6)
    ax1.plot(x, f1, '-k')
    ax1.set_xlim(0, x[0])
    ax1.set_ylim(60,200)
    ax1.set_ylabel('F10.7 [sfu]')
    ax1.grid(linestyle=':', color='gray')

    ax2.plot(x, x*0+160, '-k')
    y0 = t0.utc_datetime().year
    yn = swlast.utc_datetime().year + 1
    for y in range(y0, yn):
        ta = tsc.utc(y, 1, 1).tt - t0.tt
        tb = tsc.utc(y, 7, 1).tt - t0.tt
        ax2.plot(ta, 160, '|k')
        ax2.text(tb, 165, str(y), va='bottom', ha='center', color='k',
                 fontsize='small')
        
    ax2.plot(x, ap, '-', color='salmon', lw=0.6)
    ax2.set_xlim(0, x[0])
    ax2.set_xlabel('Days since {:}'.format(t0.utc_iso()[0:10]))
    ax2.set_ylabel('Avg. Ap')
    ax2.grid(linestyle=':', color='gray')
    
    plt.tight_layout()
    plt.savefig('data/SWdata.png')
    plt.savefig('data/SWdata.pdf')
    plt.show()
    
    return