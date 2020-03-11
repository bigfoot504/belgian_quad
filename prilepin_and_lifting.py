# 3 Oct 2019 Created
# 29 Jan 2020 Added to atom/python/chromebook

import numpy as np
import matplotlib.pyplot as plt

# 55-65%, 3-6 reps/set, 24 opt reps, 18-30 rep triangle
# 70-80%, 3-6 reps/set, 18 opt reps, 12-24 rep triangle
# 80-90%, 2-4 reps/set, 15 opt reps, 10-20 rep triangle
#   90%+, 1-2 reps/set,  7 opt reps,  4-10 rep triangle

lper = nparray([55,70,80,90]);  lvol = np.array([18,12,10,4])
mper = nparray([65,75,85,95]);  mvol = np.array([24,18,15,7])
uper = nparray([65,80,90,100]); uvol = np.array([30,24,20,10])
plt.plot(lvol,lper,'bo',lvol,lper,'b-') # lower extreme
plt.plot(mvol,mper,'ro',mvol,mper,'r-') # middle extreme
plt.plot(uvol,uper,'go',uvol,uper,'g-') # upper extreme
plt.plot(lvol,uper,'b--',uvol,lper,'g--') #
plt.ylabel('Intensity %')
plt.xlabel('Volume')
plt.show()

# Show same thing but with weight added
sqwt = 500
slwt = sqwt*lper/100; sltn = slwt*lvol
smwt = sqwt*mper/100; smtn = smwt*mvol
suwt = sqwt*uper/100; sutn = suwt*uvol
plt.plot(smtn,smwt,'ro',smtn,smwt,'r-')
plt.plot(sltn,slwt,'bo',sltn,slwt,'r-')
plt.plot(sutn,suwt,'go',sutn,suwt,'r-')
plt.plot(sltn,suwt,'b--',sutn,slwt,'r-')
plt.ylabel('Weight')
plt.xlabel('Tonnage')
plt.title('Squat Weight vs Tonnage at 1RM %1.0f' % sqwt)
plt.show()
