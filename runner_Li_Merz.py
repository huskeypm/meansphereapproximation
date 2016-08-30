import cPickle as pickle
import numpy as np
from timeit import default_timer as timer

class Params():   
  def __init__(self,pklName="data.pkl"):   #pklName=None):
    #self.pklName = pklName   # resulting pklName name 
    self.idxOxy = 0
    self.idxCl = 1
    #self.idxNa = 2
    self.idxCa = 3
    self.idxK  = 2
    self.idxMg = 4
    self.indices=["O","Cl","Na","Ca"]  # ALWAYS put oxygen first
    self.ref_conc_M = np.array([1e-200, (100.0e-3), 100.0e-3, 0])  # [M] order is  O, Cl, Na, Ca (bath) 
    self.zs = np.array([-0.5, -1.0, 1.0, 2.00]) #input charges here
    self.sigmas = np.array([0.278, 0.362, 0.204, 0.320])
    self.cacl2 = 1e-6

    self.V_i = 0.375 # nm^3
    self.filter_dielectric= 63.5
    self.nOxy = 8.0
    self.isSetup = False
    self.pklName = pklName

  def setup(self,nOxy=None,cacl2=None):
    if nOxy==None:
      nOxy = self.nOxy
    else: 
      self.nOxy = nOxy
    if cacl2==None:
      cacl2= self.cacl2 
    else: 
      self.cacl2 = cacl2

    # set ion concentrations 
    self.nIons = np.shape(self.ref_conc_M)[0]
    self.Ns = np.ones( self.nIons )# Number of ions in filter (first number is fixed, others are determined by MSA)
    self.Ns[self.idxOxy] = nOxy
    print "nOxy: ", nOxy
    print "Ns: ", self.Ns

    # update concs
    self.conc_M = np.copy(self.ref_conc_M)
    self.conc_M[self.idxCl] = self.ref_conc_M[self.idxCl] + 2 * self.cacl2
    self.conc_M[self.idxCa] = self.cacl2
    self.isSetup = True  


# hard coding example that includes Mg 
def test1(dielectric,nOxy):    #filterVolume,filterDielectric,nOxy): 
    print "WARNING: mg parameters are not verified!!!" 
    mgcl2 = 2e-3 # [M]
    cacl2 = 1e-6 # [M] 
    kcl = 150e-3 # [M]
    #pklName = "data_%5.3f_filter_%4.1f_dielectric_%2.1f_nOxy_Mg_added_LiMerz_TEST.pkl"%(filterVolume,filterDielectric,nOxy)
    params = Params()
    params.indices=["O","Cl","K","Ca","Mg"]  # ALWAYS put oxygen first
    # NOT THE MOST INTUITIVE, BUT WE ADD CACL2 VIA SETUP()
    params.ref_conc_M = np.array([1e-200, 
                                (kcl + 2*mgcl2), 
                                 kcl,
                                 0.0,             
                                 mgcl2])  # [M] order is  O, Cl, K, Ca, Mg     
    params.zs = np.array([-0.5, -1.0, 1.0, 2.00, 2.00]) #input charges here
    params.sigmas = np.array([0.278, 0.454, 0.35275, 0.32, 0.28])
    params.filter_dielectric = dielectric
    params.nOxy = nOxy

    params.setup(nOxy,cacl2)

    #daLoop(params = params,volumes=np.linspace(0.375,0.375,1))
    daLoop(params = params,volumes=np.linspace(0.375,1.000,8))#5))
    
    print "Finished with runs"

def debug(): 
    print "WARNING: mg parameters are not verified!!!" 
    params = Params()
    params.indices=["O","Cl","Na","Ca","Mg"]  # ALWAYS put oxygen first

    ##
    ## Equivalent to validation case, except have entry for mg 
    ##
    if 0:
      nOxy = 8
      mgcl2 = 0 # [M]
      cacl2 = 0    # [M] 
      kcl = 100e-3 # [M]
      params.indices = ["O","Cl","K","Ca","Mg"]
      params.ref_conc_M = np.array([1e-200, kcl+2*mgcl2, kcl, cacl2,mgcl2])  # [M] order is  O, Cl, Na, Ca, mg (bath) 
      params.zs = np.array([-0.5, -1.0, 1.0, 2.00,2.0]) #input charges here
      params.sigmas = np.array([0.278, 0.362, 0.204, 0.320,0.28])
      params.setup(nOxy,cacl2)
      daLoop(params = params,volumes=np.linspace(0.375,1.000,5))
      # AGREES W VALIDATION at V=0.375, stable otherwise  
    
    ##
    ## Adding in mgcl2 
    ##
    if 0: 
      nOxy = 8
      mgcl2 = 2e-3 # [M]
      cacl2 = 1e-6 # [M] 
      kcl = 100e-3 # [M]
      params.indices = ["O","Cl","K","Ca","Mg"]
      params.ref_conc_M = np.array([1e-200, kcl+2*mgcl2, kcl, 0,mgcl2])  # [M] order is  O, Cl, Na, Ca, mg (bath) 
      params.zs = np.array([-0.5, -1.0, 1.0, 2.00,2.0]) #input charges here
      params.sigmas = np.array([0.278, 0.362, 0.204, 0.320,0.28])
      params.setup(nOxy,cacl2)
      daLoop(params = params,volumes=np.linspace(0.375,1.000,5))

    ##
    ## Dielec
    ## 
    if 0: 
      nOxy = 8
      mgcl2 = 2e-3 # [M]
      cacl2 = 1e-6 # [M] 
      kcl = 100e-3 # [M]
      params.indices = ["O","Cl","K","Ca","Mg"]
      params.filter_dielectric = 40.
      params.ref_conc_M = np.array([1e-200, kcl+2*mgcl2, kcl, 0,mgcl2])  # [M] order is  O, Cl, Na, Ca, mg (bath) 
      params.zs = np.array([-0.5, -1.0, 1.0, 2.00,2.0]) #input charges here
      params.sigmas = np.array([0.278, 0.362, 0.204, 0.320,0.28])
      params.setup(nOxy,cacl2)
      daLoop(params = params,volumes=np.linspace(0.375,1.000,5))

    ##
    ## Noxy   
    ## 
    if 1: 
      nOxy = 6.0
      mgcl2 = 2e-3 # [M]
      cacl2 = 1e-6 # [M] 
      kcl = 100e-3 # [M]
      params.indices = ["O","Cl","K","Ca","Mg"]
      params.filter_dielectric = 40.
      params.ref_conc_M = np.array([1e-200, kcl+2*mgcl2, kcl, 0,mgcl2])  # [M] order is  O, Cl, Na, Ca, mg (bath) 
      params.zs = np.array([-0.5, -1.0, 1.0, 2.00,2.0]) #input charges here
      params.sigmas = np.array([0.278, 0.362, 0.204, 0.320,0.28])
      params.setup(nOxy,cacl2)
      daLoop(params = params,volumes=np.linspace(0.375,1.000,30))

    # debug iter 
    if 0: 
      nOxy = 6.0
      mgcl2 = 2e-3 # [M]
      cacl2 = 1e-6 # [M] 
      kcl = 100e-3 # [M]
      params.indices = ["O","Cl","K","Ca","Mg"]
      params.filter_dielectric = 40.
      params.ref_conc_M = np.array([1e-200, kcl+2*mgcl2, kcl, 0,mgcl2])  # [M] order is  O, Cl, Na, Ca, mg (bath) 
      params.zs = np.array([-0.5, -1.0, 1.0, 2.00,2.0]) #input charges here
      params.sigmas = np.array([0.278, 0.362, 0.204, 0.320,0.28])
      params.setup(nOxy,cacl2)
      #daLoop(params = params,volumes=np.linspace(0.375,1.000,5))
      # Inserted from where loop failed 
      muiexsPrev = np.array([ 0.35834118,-0.6872931, -1.49702449,-7.09666043,-7.31364988])
      psiPrev = -16.1501858134
      params.V_i = 0.948
      daIter(
        params=params,
        muiexsPrev=muiexsPrev,psiPrev=psiPrev)

      
       
def daLoop(
    noRun=False,
    volumes = np.linspace(0.2,1.5,66),
    params=None
    ): 
    #print "Oxys: ", nOxys
    print "Volumes: ", volumes
    #print "Dielectrics: ", dielectrics
    
    #Oxy_len = len(nOxys)
    #dielectrics_len = len(dielectrics)
    vol_len = len(volumes)
    total_counter = vol_len #float(Oxy_len * dielectrics_len * vol_len)

    #for i in len(nOxys):
    #  for j in len(dielectrics):
    #    for k in len(vol):
    #      total_counter += 1
    print "Total runs will be: ", total_counter

    # inputs
    counter = 0.0

    #for i,nOxy in enumerate(nOxys):
     
    # now setup
    muiexsPrev = np.zeros(params.nIons)
    psiPrev = 0.0

    # for j,dielectric in enumerate(dielectrics):
    for k,vol in enumerate(volumes):
 
      #print "Oxy: ", nOxy
      #print "dielc: ", dielectric
      #print "Vol: ", vol
   
      params.V_i = vol
      params.pklName = "data_%5.3f_filter_%4.1f_dielectric_%2.1f_nOxy_Mg_added_LiMerz.pkl"%\
        (params.V_i, params.filter_dielectric, params.nOxy)
      if noRun:
        return params.pklName 

      daIter(noRun=noRun,
        params=params,
        muiexsPrev=muiexsPrev,psiPrev=psiPrev)


      # read info
      results = ReadPickle(params.pklName)
      daParams = results["params"]
      print daParams.conc_M
      muiexsPrev=results["muFilter"]    
      psiPrev = results["donnanPotential"]

      counter += 1.0
      percentage_done = round((counter/total_counter) * 100,2)
      print "Current job is", counter, "out of", total_counter, " vol %f"%vol
      print percentage_done,"% done"


# Runs single iteration 
def daIter(
    noRun=False,
    params = None,
    alpha = 0.0010,
    muiexsPrev=None,
    psiPrev = None
    ): 
    # if parameters object not passed, create one and initialize it
    if params==None:
      params = Params()
      params.setup(params.nOxy,params.cacl2)

    # check that params was set up
    assert(params.isSetup), "Need to call params.setup() before proceeding"

    # initialize solver with solutions from prior run 
    if muiexsPrev==None:
      muiexsPrev = np.zeros(params.nIons)
      psiPrev = 0.0

    if noRun:
      return params.pklName 
    
    # outputs 
    results = {}
    start = timer()
    
    # run MSA 
    import MSAtools as msa
    mufilteri,donnanPotentiali,mu_ESi,mu_HSi,rhoFilteri = msa.SolveMSAEquations(
      params.filter_dielectric,
      params.conc_M,
      params.zs,
      params.Ns,
      params.V_i,
      params.sigmas,
      params.nOxy,                                                 
      psiPrev=psiPrev,
      muiexsPrev=muiexsPrev,
      alpha=alpha, 
      verbose=False)
      #verbose=True)

    ## store results     
    # returned
    results["muFilter"] = mufilteri
    results["donnanPotential"] = donnanPotentiali
    results["mu_ES"] = mu_ESi
    results["mu_HS"] = mu_HSi
    results["rhoFilter"] = rhoFilteri
    results["params"] = params 

    # report information 
    print params.indices
    print "mufilteri",mufilteri
    print "rhoFilter",rhoFilteri
    print "donnanPotential",donnanPotentiali
    end = timer()
    print(end - start), " elapsed seconds "

    # write info
    output = open(params.pklName, 'wb')
    pickle.dump(results, output)
    output.close()
    print "Printed ", params.pklName

    return params.pklName


def ReadPickle(pklName): 
     output = open(pklName, 'rb')
     results = pickle.load(output)
     output.close()
     return results 

def validation():
# #filterVolume=0.375, filterDielectric=40.0,nOxy=8.0,noRun=False,params=None):
      params = Params()
      nOxy=8; cacl2=1e-6
      params.setup(nOxy,cacl2) 
      pklName = daIter(params=params)
      results = ReadPickle(pklName)
      # Donnan
      psiPrev = results["donnanPotential"]  
      val_160830 = -172.558
      assert(np.abs ( psiPrev - val_160830 ) < 0.001 ),\
            "FAIL! Curr: %f"%psiPrev

      # chemical potentials 
      ions = ["Cl","Na","Ca"] 
      vals_Nonner= [6.,0.5,-4.] # kT, Fig 2 
      muFilter= results["muFilter"]  
      for i, ion in enumerate(ions):  
        j = params.indices.index(ion)
        mui = muFilter[j]
        muref = vals_Nonner[i]
        if (np.abs ( mui- muref) > 0.001 ):
          print "FAIL! %s Curr: %f/ Ref: %f"% (ion,mui,muref)

        #assert(np.abs ( mui- muref) < 0.001 ),\
        #    "FAIL! %s Curr: %f/ Ref: %f"% (ion,mui,muref)

# default 
if __name__ == "__main__":
  import sys
  msg = """ Run as runner.py filterVol filterDielec"""

  if len(sys.argv) < 2: # 3 and sys.argv[1]!="-validation":
      raise RuntimeError(msg)

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-nOxy"):              
      nOxy = np.float( sys.argv[i+1] ) 
    if(arg=="-validation"): 
      validation()
      quit()
    if(arg=="-debug"): 
      debug()
      quit()
    if(arg=="-test1"): 
      #test1(np.float(sys.argv[1]),np.float(sys.argv[2]),np.float(sys.argv[3]))
      test1(np.float(sys.argv[1]),np.float(sys.argv[2]))
      quit()


  daIter()      #np.float(sys.argv[1]),np.float(sys.argv[2]),np.float(sys.argv[3]))

