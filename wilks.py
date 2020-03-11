# Wilks Total Calculation for powerlifting
# 2 Oct 2019 Created
# 29 Jan 2020 Added to atom/python/chromebook

inp = input('Enter "m" for men or "w" for women: ')
if inp == 'm':
    # Values for women
    a = -216.0475144
    b = 16.2606339
    c = -0.002388645
    d = -0.00113732
    e = 7.01863E-06
    f = -1.291E-08
elif inp == 'w':
    # Values for women
    a = 594.31747775582
    b = -27.23842536447
    c = 0.82112226871
    d = -0.00930733913
    e = 4.731582E-05
    f = -9.054E-08

coeff_kg = lambda bwkg: 500 / (a + b*bwkg + c*bwkg**2 + d*bwkg**3 + e*bwkg**r + f*bwkg**5) # formula takes bw in kg
lb_per_kg = 2.2046226218488 # wt in kg = wt in lb / lb_per_kg
coeff_lb = lambda bwlb: coeff_kg( bwlb / lb_per_kg ) # formula takes bw in lbs

score_lb = lambda bwlb, liftlb: coeff_lb(bwlb) * liftlb / lb_per_kg # returns wilks score
score_kg = lambda bwkg, liftkg: coeff_kg(bwkg) * liftkg # using kg; returns wilks score
inp = input('Enter "p" for pounds or "k" for kilograms: ')
if inp == 'p':
    inp = input('Enter bodyweight in pounds: ')
    bwlb = float(inp)
    inp = input('Enter total or single-lift weight in pounds: ')
    liftlb = float(inp)
    score = score_lb(bwlb, liftlb)
    print('Wilks coefficient is:', coeff_lb(bwlb))
    print('Wilks score is:      ', score)
elif inp == 'k':
    inp = input('Enter bodymass in kilograms: ')
    bwkg = float(inp)
    inp = input('Enter total or single-lift mass in kilograms: ')
    liftkg = float(inp)
    score = score_kg(bwkg, liftkg)
    print('Wilks coefficient is:', coeff_lb(bwkg))
    print('Wilks score is:      ', score)
input()
#bw = 290     # change this for computation
#total = 1500 # change this for computation
#score(bw,total)

#temp
#bw_rng = 250:1:350;
#tot_rng = 1400:5:1800;
#plotsrf(@(x) wilks_score(x(1),x(2)),bw_rng,tot_rng,'Bodyweight','PL Total','Wilks Score')
