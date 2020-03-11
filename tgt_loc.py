# 16 Oct 2019 Created
# 29 Jan 2020 Added to atom/python/chromebook

import numpy as np
import matplotlib.pyplot as plt

def quit_msg():
    print('\n(Enter "q" to quit.)')
def error_msg():
    print('\n\n***INPUT ERROR***\n\n')

while True:
    quit_msg()
    inp = input('Enter 10-digit grid (e.g., "12345 67890"): ') # try 23414 43278
    if inp == 'q': break
    try:
        inp = inp.split()
        if not (inp[0].isdigit() and len(inp[0])==5 and inp[1].isdigit() and len(inp[1])==5):
            print('\n\n***ERROR: Must be five digits, a space, and five digits.')
            print('(e.g., "12345 67890")***\n\n'); continue
    except: # if it can't split
        print('\n\n***ERROR: Must be five digits, a space, and five digits.')
        print('(e.g., "12345 67890")***\n\n'); continue
    myE = float(inp[0])
    myN = float(inp[1])
    quit_msg()
    inp = input('Enter direction to target (3-digit for degrees or 4-digit for mils): ')
    if inp == 'q': break
    try:
        if len(inp) <= 3:
            if float(inp) < 0 or float(inp) > 360:
                print('\n\n***ERROR: Invalid degrees range. (Must be 0-360.)***\n\n'); continue
            dirr = float(inp) / 360 * 2*np.pi # degrees to radians
        elif len(inp) == 4:
            if float(inp) < 0 or float(inp) > 6400:
                print('\n\n***ERROR: Invalid mils range. (Must be 0-6400.)***\n\n'); continue
            dirr = float(inp) / 6400 * 2*np.pi # mils to radians
    except: error_msg()
    quit_msg()
    print('G-M angle (in degrees):')
    print('Enter a negative number (-) if magnetic north is WEST of grid north.')
    print('Enter a positive number (+) if magnetic north is EAST of grid north.')
    print('Enter 0 if direction is already a grid direction.')
    inp = input('Enter G-M angle (in degrees): ')
    if inp == 'q': break
    try:
        if float(inp) <= -90 or float(inp) >= 90:
            print('\n\n***ERROR: G-M angle must be exclusively between -90 to 90.***\n\n')
            continue
        gm = float(inp)
    except:
        print('\n\n***ERROR: G-M angle must be exclusively between -90 to 90.***\n\n')
        continue
    dirr = dirr + gm / 360 * 2*np.pi
    quit_msg()
    inp = input('Enter distance in meters: ')
    if inp == 'q': break
    try:
        if float(inp) < 0:
            print('\n\n***ERROR: Distance must be a non-negative number.***\n\n')
            continue
        DC = True if float(inp) < 600 else False
        dist = float(inp)
    except:
        error_msg(); continue
    # compute target easting
    enE = np.int(np.round(myE + np.sin(dirr) * dist))
    if enE < 0:
        print('\n\n***ERROR: Cannot provide target grid, because target is too')
        print('far WEST (in another grid zone designator).***\n\n')
        continue
    if enE > 99999:
        print('\n\n***ERROR: Cannot provide target grid, because target is too')
        print('far EAST (in another grid zone designator).***\n\n')
        continue
    if   enE < 10:    enEs = '0000' + str(enE)
    elif enE < 100:   enEs = '000'  + str(enE)
    elif enE < 1000:  enEs = '00'   + str(enE)
    elif enE < 10000: enEs = '0'    + str(enE)
    else: enEs = str(enE)
    # compute target northing
    enN = np.int(np.round(myN + np.cos(dirr) * dist))
    if enN < 0:
        print('\n\n***ERROR: Cannot provide target grid, because target is too')
        print('far SOUTH (in another grid zone designator).***\n\n')
        continue
    if enN > 99999:
        print('\n\n***ERROR: Cannot provide target grid, because target is too')
        print('far NORTH (in another grid zone designator).***\n\n')
        continue
    if   enN < 10:    enNs = '0000' + str(enN)
    elif enN < 100:   enNs = '000'  + str(enN)
    elif enN < 1000:  enNs = '00'   + str(enN)
    elif enN < 10000: enNs = '0'    + str(enN)
    else: enNs = str(enN)

    # plot results
    x = np.linspace(-dist,dist,101) + myE
    y = np.sqrt(dist**2 - (x-myE)**2) + myN
    plt.plot(x,y); plt.plot(x,2*myN-y); #plt.show()
    plt.plot([myE, enE], [myN, enN])
    plt.show()

    print('Target Location: ', enEs, enNs)
    if DC == True:
        print('***WARNING: Your target may be within DANGER CLOSE proximity depending upon the munition.***')

    inp = input('Hit "Enter" to do another target or "q" to quit: ')
    if inp == 'q': break
    print('\n\n')
