import numpy as np
from numba import jit
#
 
N = 6.022e23 # #/L
L_to_mL = 1e-3
mL_to_nm3 = (1e-7)**3 # 1/cm^3 * (cm/nm)**3
M_to_N_o_nm3  = L_to_mL * mL_to_nm3 * N
kT_mV = 25.6 # mV * e0
kT_kcalmol = 0.59 # [kcal/mol]
m_to_nm = 1e9
idxOxy=0
e0 = 1/kT_mV
print "WARNING: oxy always must be first entry" 



# for packaging results
results = {}
results["e0"] = e0


# lambda = e^2/ (4 pi eps eps_r kT)
# numbers from Israelachvili
# Verified 
#@jit
def CalcBjerrum(epsilon=78.4):
  denom = 8.854e-12 * epsilon *1.381e-23 * 298 # eps0*eps*k*T
  num = (1.602e-19)**2 # e^2

  lambda_b = num/(4*np.pi*denom)
  #print "Bj. len [nm] %f"%(lambda_b*1e9)
  return lambda_b


# verified (see notebook) 
#@jit
def CalcRhoFilter(muBath, muiexPrev, psi, zs, rhoisBath):
    #raise RuntimeError("See me!!!") 
    #noOxyRhoiBath = [1,rhoibath[iOxy:]]  # assume first oxy is 1 for numerical stability 
    rhoisBath[idxOxy]= 1.0e-200  # assume first oxy is 1 for numerical stabilit
   # print np.shape(muBath), np.shape(muiexPrev)
    deltaG = -(muBath) + zs*e0*psi + (muiexPrev) # from Nonner
    results["deltaG"]= deltaG
    #print muBath, zs*e0*psi, muiexPrev
   # print np.shape(muBath) ,np.shape(muiexPrev)
    kTunits = 1.
    #print "deltaG [kT] ", deltaG
    # for kT=1, kT ln p' = kT ln p - deltaG ---> p' = p exp(-deltaG/kT)
    pFilter = np.exp(-deltaG/kTunits)*rhoisBath
    rhoisBath[idxOxy]= 0. 
    return pFilter

# verified (see notebook) 
#@jit
def CalcRhoFilterWTF(muBath, muiexPrev, psi, zs, rhoisBath):
    #noOxyRhoiBath = [1,rhoibath[iOxy:]]  # assume first oxy is 1 for numerical stability 
    rhoisBath[idxOxy]= 1.0e-200  # assume first oxy is 1 for numerical stabilit
   # print np.shape(muBath), np.shape(muiexPrev)
    #array of solvation energies... converted to kT in the delmu_born eqn after
    muw = np.array([0,-304,-351,-1608,-1931.4])
    #eqn. 9 from Nonner 2001 JPC B
    delmu_born = (muw / 0.59) * (25.0-78.4)/( 25.0 * ( 78.4 - 1) )
    deltaG = -(muBath) + zs*e0*psi + (muiexPrev) - delmu_born # from Nonner
    #print muBath, zs*e0*psi, muiexPrev
   # print np.shape(muBath) ,np.shape(muiexPrev)
    kTunits = 1.
    #print "deltaG [kT] ", deltaG
    # for kT=1, kT ln p' = kT ln p - deltaG ---> p' = p exp(-deltaG/kT)
    pFilter = np.exp(-deltaG/kTunits)*rhoisBath
    rhoisBath[idxOxy]= 0. 
    return pFilter
    
# Verified 
#@jit
def CalcKappa(conc_M):
  ionSum = np.sum(conc_M) 
  kappaSqd = 6.022e26 * ionSum * CalcBjerrum() * 4 * np.pi
  return np.sqrt(kappaSqd)

#zs = np.array([-0.5, -1.0, 1.0, 2.0, 2.0])
#@jit
def CalcMuexs(Gamma,zs,lambda_b=CalcBjerrum()):
    #for i in zs:
    muiexs = -kT_kcalmol*lambda_b * m_to_nm  
    muiexs *= Gamma * zs*zs
   
    #print lambda_b*m_to_nm, Gamma
    #print lambda_b*m_to_nm* Gamma
    #print lambda_b*m_to_nm* Gamma*kT

    return muiexs

#@jit
def UpdatePsi(rhos,zs,psiPrev):
    
    numpsi = np.sum(zs*rhos)
    denompsi = np.sum(rhos*zs*zs)
    delpsi = numpsi/denompsi
    psi = psiPrev + delpsi
    return psi




# x[0] Gamma
# Ns: number per nm^3  [#]  !! not [#/nm^3] since earlier multiplied by V [nm^3]
# Vs: V [nm^3]
def MSAeqn(x,Ns, Vs,zs,sigmas,lambda_b=CalcBjerrum(),eta=0,verbose=False):  # eqn 7, nonner
  ## skip evaluating eta based on |eta*sigma^2|<0.04arg. 
  ## skip evaluating omega since ignoring eta

  ## Solve for Gamma in eqn 7   
  # evaluate summation in eqn 7, nonner
  Gamma = x[0]  
  sqdTerm = (Ns/Vs)*((zs - eta*sigmas*sigmas)/(1 + Gamma*sigmas ))**2
  # hack 
  #sqdTerm = Ns * zs**2  # ONLY for sigma=0
  #sqdTerm*=Ns*sqdTerm/Vs
  sqdTerm = np.sum(sqdTerm) 
   
  #print "sqd ", sqdTerm  
#  rho = N_i/V_i  # [#/nm^3 --> M]
#  rho = # HACK   
  # for now, assume that we have only two ions
  #rho_M = np.sum(Ns/Vs) * convFactor 
  #print "rho [M] ", rho_M 
     
  # Bj. length contains 1/4pi, whereas eqn 7 does not, therefore we multiply Bj length by 4pi
  # something funky here - need to show that kappa^2 = (sqrt(rho[M])/0.304)^2 = 4 gamma^2
  fourGammaSqd = 4*np.pi * lambda_b * sqdTerm 
  fourGammaSqd*= 1e9   # M_to_nM  
  
  
  # for debugging. should get debyelength back if sigma =0.
  #debyeLength = 1/np.sqrt(fourGammaSqd)
  #print "Mr. Debye: %f [nm]" % debyeLength  
    
  residual =  4*Gamma*Gamma - fourGammaSqd 
  objective = residual*residual
  if verbose:  
    print "FGS ", fourGammaSqd      
    print "LHS %f RHS %f lastGamma %f residual %f" % (4*Gamma*Gamma,fourGammaSqd,Gamma, residual)
  return objective
#print Ns/Vs




#@jit
def deltaphiHS(xi_0, xi_1, xi_2, xi_3,delta):
    HSphi = xi_3/delta + (3.*xi_1*xi_2)/(xi_0*delta*delta) + (3.*xi_2*xi_2*xi_2)/(xi_0*delta*delta*delta)
    return HSphi



## now we calculate the HS chemical potential

@jit
def HSmuis(sigmas, delta, xi_0,xi_1, xi_2,xi_3,HSphifilter):
    HSmu = (3.*xi_2*sigmas + 3.*xi_1*sigmas*sigmas)/delta 
    HSmu+= (9.*xi_2*xi_2*sigmas*sigmas)/(2.*delta*delta) 
    HSmu+= xi_0*sigmas*sigmas*sigmas*(1.+HSphifilter) - np.log(delta)
    return HSmu

## first we define a xi function
#@jit
@jit
def xis(rhos,sigmas,n):
    xivalue = (np.pi / 6.) * np.sum(rhos * (sigmas**n))
    return xivalue

@jit
def CalcHS(rhosfilter,sigmas):
  #print rhosfilter
  xi_3 = xis(rhosfilter, sigmas, 3)
  delta = 1. - xi_3
  xi_1 = xis(rhosfilter, sigmas, 1)
  xi_2 = xis(rhosfilter, sigmas, 2)
  xi_0 = xis(rhosfilter, sigmas, 0)
  #print "xi_n (n= 0 to 3) =", xi_0, xi_1, xi_2, xi_3; 
  #print "delta =", delta

## this gives us all the xi components we need to calculate the rest of the HS stuff


  HSphibath = deltaphiHS(xi_0, xi_1, xi_2,xi_3,delta)
  #print HSphibath

  return HSmuis(sigmas, delta, xi_0, xi_1, xi_2,xi_3, HSphibath)



@jit 
def getdeltaes(rhos,sigmas):
    sumtermdeltaes = np.sum(rhos*sigmas)
    pitermdeltaes = (np.pi*sumtermdeltaes)/6.
    deltaes = 1. - pitermdeltaes
    return deltaes


@jit
def getomegaes(Gamma,deltaes,rhos,sigmas):
    sumtermomegaes = np.sum((rhos*(sigmas**3))/(1.+(Gamma*sigmas)))
    pitermomegaes = (np.pi * sumtermomegaes) / (2.*deltaes)
    omegaes = 1. + pitermomegaes
    #print sumtermomegaes
    #print pitermomegaes
    return omegaes


@jit
def getetaes(omegaes,deltaes,rhos,sigmas,zs,Gamma):
    sumtermetaes = np.sum((rhos*sigmas*zs)/(1.+(Gamma*sigmas)))
    pitermetaes = (np.pi * sumtermetaes)/(2.*deltaes)
    etaes = pitermetaes/omegaes
    #print sumtermetaes
    #print pitermetaes
    return etaes

@jit
def getgamma(rhos,zs,sigmas,etaes,GammaFilterPrev ,lambda_b=CalcBjerrum()):
    sqdTerm = (rhos)*((zs - etaes*sigmas*sigmas)/(1 + GammaFilterPrev*sigmas ))**2
    sqdTerm = np.sum(sqdTerm)
    fourGammaSqd = 4*np.pi * lambda_b * sqdTerm
    fourGammaSqd*= 1e9
    newGamma = np.sqrt(fourGammaSqd) / 2.
    return newGamma


def test():
  mufilteri,donnanPotentiali,mu_ESi,mu_HSi,rhoFilteri = SolveMSAEquations(
    63,
    np.array([1e-200, (100.0e-3), 100.0e-3, 0]),
    np.array([-0.5, -1.0, 1.0, 2.00]),
    np.array([7, 1., 1., 1.]),
    0.375,    
    np.array([0.278, 0.362, 0.204, 0.200]),
    7                
    )
  
def xiStuff(rhos,sigmas):
    xi_3_= xis(rhos, sigmas, 3)
    delta_= 1. - xi_3_
    xi_1_= xis(rhos, sigmas, 1)
    xi_2_= xis(rhos, sigmas, 2)
    xi_0_ = xis(rhos, sigmas, 0)
    #print "xi_n (n= 0 to 3) =", xi_0_, xi_1_, xi_2_, xi_3_; 
    #print "delta =", delta_
    return xi_0_, xi_1_, xi_2_, xi_3_, delta_




def SolveMSAEquations(epsilonFilter,conc_M,zs,Ns,V_i,sigmas,
  nOxy=8,   # From Nonner 
  muiexsPrev = 0.,
  maxiters = 2e4, # max iteration before loop leaves in dispair 
#  maxiters = 1e10, # max iteration before loop leaves in dispair 
  psiPrev = 0.,
  psitol = 1.0e-8,
  gammaTol = 1.0e-4, # PKH changes this (I think 1e-8 is too restrictive) 
  alpha = 1e-3,  # convergence rate (faster values accelerate convergence) 
  #maxitersGamma = 1e8, # max iteration before loop leaves in dispair 
  maxitersGamma = 1e2, # max iteration before loop leaves in dispair 
  useSelfConsistGammaOpt = True,  
  verbose=False):
  psiDiff  = 1e9
  muiexDiff = 1e9


  ## Convert concs/numbers into densities 
  conc_M_to_N_o_nm3 = conc_M * M_to_N_o_nm3 # [#/nm3]
  #!!!print conc_M_to_N_o_nm3
  rhoisBath = conc_M_to_N_o_nm3

  # enforce nOxy
  Ns[idxOxy] = nOxy
  rhoisFilter = np.array(Ns / V_i)

   # Compute debye length 
  #kappainv = 0.304 / np.sqrt(conc_M[idxNa]) # Israelach
  #print "Analy kinv [nm] %f (nacl only)" %(kappainv) # for 100 mM
  kappa = CalcKappa(conc_M) # np.sqrt(4*np.pi *msa.lambda_b * (2*6.022e26* conc_M[idxNa]))

  #!!!print "Est kinv [nm] %f " % (1e9/kappa)
  #print kappa

  ## compute gamma for bath 
  GammaBath = kappa/2*(1/m_to_nm)
  muexBath_kT = CalcMuexs(GammaBath,zs)/kT_kcalmol

  # Validated (with my notes)
  #print "mu_Na,bath [kT] " , muexBath_kT[idxNa]
  #!!!print "muexBath_kT ", muexBath_kT


  ## Calculate Bjerrum, filter chem potential 

  #GammaFilter = 2.11 # [nm] 
  # verified
  #lambdaBFilter = msa.CalcBjerrum(epsilon=epsilonFilter)

  xi_0_filter, xi_1_filter, xi_2_filter, xi_3_filter, delta_filter = xiStuff(rhoisFilter,sigmas)
  #print "xi_n (n= 0 to 3) =", xi_0_filter, xi_1_filter, xi_2_filter, xi_3_filter
  #print "delta =", delta_filter
  xi_0_bath, xi_1_bath, xi_2_bath, xi_3_bath, delta_bath = xiStuff(rhoisBath,sigmas)
  #print "xi_n (n= 0 to 3) =", xi_0_bath, xi_1_bath, xi_2_bath, xi_3_bath
  #print "delta =", delta_bath

  HSphifilter = deltaphiHS(xi_0_filter, xi_1_filter, xi_2_filter,xi_3_filter,delta_filter)
  #print HSphifilter

  HSphibath = deltaphiHS(xi_0_bath, xi_1_bath, xi_2_bath,xi_3_bath,delta_bath)


  ## Compute HS contribution to chem potential for filter and bath 
  mu_HS_filter = HSmuis(sigmas, delta_filter, xi_0_filter, xi_1_filter, xi_2_filter,xi_3_filter, HSphifilter)
  mu_HS_bath = HSmuis(sigmas, delta_bath, xi_0_bath, xi_1_bath, xi_2_bath, xi_3_bath, HSphibath)
  muexBath_kT += mu_HS_bath
  #print "muexBath_kT ", muexBath_kT


  nIons = np.shape(conc_M)
  Vs = np.ones(nIons)*V_i
  lambdaBFilter = CalcBjerrum(epsilon=epsilonFilter)
  import scipy.optimize
  iters = 0
  #for j in np.arange(maxiters):    
  while(abs(psiDiff) > psitol):
    #print psiDiff, psitol

    ##  Get Updated Rhois              
    rhoisFilter = CalcRhoFilter(muexBath_kT,muiexsPrev,psiPrev,zs,rhoisBath)
    # reset rhoisFilter[Oxy] back to original value 
    rhoisFilter[idxOxy] = nOxy/V_i
    if verbose:
      print "p [M] (Before  oxy correct): ", rhoisFilter
      print "p [M] (After oxy correct): ", rhoisFilter
        
    ##!GammaFilter = msa.CalcKappa(rhoisFilter)/2.  # use Gamma based on filter concs
    ##!GammaFilter/= msa.m_to_nm
    #print GammaFilter
    GammaFilterPrev = CalcKappa(rhoisFilter)/2.  # use Gamma based on filter concs
    GammaFilterPrev /= m_to_nm
    itersgamma = 0
    if useSelfConsistGammaOpt:
      #print "Going into selfconsist gamma opt"
      gammadiff = gammaTol + 5  # force at least one iteration 
      while(abs(gammadiff) > gammaTol):
       
        deltaes = getdeltaes(rhoisFilter,sigmas)
        omegaes = getomegaes(GammaFilterPrev,deltaes,rhoisFilter,sigmas)
        etaes = getetaes(omegaes,deltaes,rhoisFilter,sigmas,zs,GammaFilterPrev)
        GammaFilter = getgamma(rhoisFilter,zs,sigmas,etaes,GammaFilterPrev,CalcBjerrum())
        gammadiff = (GammaFilterPrev - GammaFilter)/GammaFilterPrev

        itersgamma += 1
        if itersgamma > maxitersGamma:
            print "Your function broke (itersgamma %d exceeded, prev/curr %f/%f/%f)"%\
                   (maxitersGamma,GammaFilterPrev, GammaFilter,gammadiff)
            break
        GammaFilterPrev = GammaFilter
    # Use scipy optimize
    else:
      xopt = scipy.optimize.fmin(func=MSAeqn,x0=GammaFilter,disp=False,\
                               args=(rhoisFilter,Vs,zs,sigmas,lambdaBFilter))
    
      GammaFilter = xopt[0]

    if verbose:
      print "GammaFilter", GammaFilter
    #print "GammaF %f" % GammaFilter
      # get updated muis    
    #muiexs = np.zeros(nIons)
    #print zis


    mu_ES = CalcMuexs(GammaFilterPrev,zs,lambdaBFilter)
    mu_HS = CalcHS(rhoisFilter,sigmas)
    
    muiexs = mu_ES + mu_HS
    if verbose:
      print "mu_ES",mu_ES
      print "mu_HS",mu_HS
      print "curr muiexs " , muiexs      # relax muis
      print "prev muiexs",muiexsPrev
    #### s.b. eqn 31
    # rescale to slow convergence 
    muiexs = (alpha*muiexs + muiexsPrev)/(1.+ alpha)
    if verbose:
        print "rescaled muiexs " , muiexs

    # muiexDiff
    muiexDiff = np.sum(muiexsPrev**2)
    muiexDiff -= np.sum(muiexs**2)
    muiexDiff = np.abs(muiexDiff)

    ## Update psi
    psi = UpdatePsi(rhoisFilter,zs,psiPrev)
    psiDiff = np.abs(psiPrev-psi)/psi

    psiTol = -1
    if psi >(0-psiTol):
      print "Psi ", psi 
      raise RuntimeError("psi should not be positive for a negatively-charged filter. Somethings wrong")
    if verbose:
      print "itersgamma %d, iters %d"%(itersgamma,iters) 
      print "psi %f/psiDiff %f"% (psi,psiDiff)
      print "rhoisFilter ", rhoisFilter
      print "muiexDiff", muiexDiff
      print "======="
    ## Update prev
    psiPrev = psi
    muiexsPrev = muiexs
    iters += 1
    if iters > maxiters:
      print "Your function broke!"
      break
  # muiexs - chem potential of filter 
  # psi donnan potential 
    ## END LOOP 
  # store results     
  results["muBath"] = muexBath_kT
  results["muFilter"] = muiexs     
  results["donnanPotential"] = psi
  results["mu_ES"] = mu_ES 
  results["mu_HS"] = mu_HS 
  results["rhoFilter"] = rhoisFilter
  print "itersgamma %d, iters %d"%(itersgamma,iters) 

  return muiexs,psi,mu_ES,mu_HS,rhoisFilter

