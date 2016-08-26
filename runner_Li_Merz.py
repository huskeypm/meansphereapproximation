import cPickle as pickle
import numpy as np
import MSAtools as msa
from timeit import default_timer as timer

class Params():   
  def __init__(self):   #pklFile=None):
    #self.pklFile = pklFile   # resulting pklFile name 
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

  def setup(self,nOxy):
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


# hard coding example that includes Mg 
def test1(dielectric,nOxy):    #filterVolume,filterDielectric,nOxy): 
    print "WARNING: mg parameters are not verified!!!" 
    mgcl2 = 2e-3 # [mM]
    #pklFile = "data_%5.3f_filter_%4.1f_dielectric_%2.1f_nOxy_Mg_added_LiMerz_TEST.pkl"%(filterVolume,filterDielectric,nOxy)
    params = Params()
    params.indices=["O","Cl","K","Ca","Mg"]  # ALWAYS put oxygen first
    params.ref_conc_M = np.array([1e-200, (150.0e-3 + 2*mgcl2), 150.0e-3, 0.0, mgcl2])  # [M] order is  O, Cl, K, Ca, Mg     
    params.zs = np.array([-0.5, -1.0, 1.0, 2.00, 2.00]) #input charges here
    params.sigmas = np.array([0.278, 0.454, 0.35275, 0.32, 0.28])

    # now setup
    #params.setup(nOxy)

    daIter(dielectric,nOxy,params = params) #filterVolume=filterVolume, filterDielectric=filterDielectric,nOxy=nOxy, params = params) 
    
    print "Finished with runs"
       
def daIter(dielectric,nOxy,noRun=False,params=None): #filterVolume=0.375, filterDielectric=40.0,nOxy=8.0,noRun=False,params=None):
    # if parameters object not passed, create one and initialize it
    if params==None:
      pklFile = "data_%5.3f_%4.1f.pkl"%(filterVolume,filterDielectric)
      params = Params(pklFile = pklFile)
      params.setup()
    
    # test
    #nOxys = np.linspace(5.0,6.0,2)
    #volumes = np.linspace(0.3,0.4,3)
    #dielectrics = np.linspace(40.,50.,2)
    
    #nOxys = np.linspace(5.0,8.0,4)
    #volumes = np.linspace(0.2,0.6,21)
    #dielectrics = np.linspace(10.0,50.0,5)

    #nOxys = np.linspace(5.0,8.0,4)
    volumes = np.linspace(0.2,1.5,66)
    #dielectrics = np.linspace(10.0,80.0,8)

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
    start = timer()
    counter = 0.0

    #for i,nOxy in enumerate(nOxys):
     
    # now setup
    params.setup(nOxy)
    muiexsPrev = np.zeros(params.nIons)
    psiPrev = 0.0
    alpha = 0.0010
     
     # for j,dielectric in enumerate(dielectrics):
    for k,vol in enumerate(volumes):
 
            #print "Oxy: ", nOxy
            #print "dielc: ", dielectric
            #print "Vol: ", vol
    
    	    pklFile = "data_%5.3f_filter_%4.1f_dielectric_%2.1f_nOxy_Mg_added_LiMerz.pkl"%(vol,dielectric,nOxy)
            if noRun:
               return pklFile 
            #print "pklfile: ", pklFile
     
            # outputs 
            results = {}
    
            # assign
            params.filter_dielectric = dielectric
            params.V_i = vol
            params.nOxy = nOxy
            #print "params.filter_dielectric: ", params.filter_dielectric
            #print "params.V_i: ", params.V_i
            #print "params.nOxy: ", params.nOxy

            # run MSA 
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

            # update previous psi/muiexs so code has better initial guess 
            psiPrev = donnanPotentiali
            muiexsPrev = mufilteri

            # report information 
            print "mufilteri",mufilteri

            end = timer()
            print(end - start), " elapsed seconds "

            # write info
            output = open(pklFile, 'wb')
            pickle.dump(results, output)
            output.close()
            print "Printed ", pklFile

            # read info
            results = ReadPickle(pklFile)
            daParams = results["params"]
            print daParams.conc_M

            counter += 1.0
            percentage_done = round((counter/total_counter) * 100,2)
            print "Current job is", counter, "out of", total_counter
            print percentage_done,"% done"

       # print "Finished dielectric: ", dielectric
      #print "Finished Oxy: ", nOxy 

def ReadPickle(pklFile): 
     output = open(pklFile, 'rb')
     results = pickle.load(output)
     output.close()
     return results 

def validation():
      pklFile = daIter()
      results = ReadPickle(pklFile)
      psiPrev = results["donnanPotential"]  
      val_160731 = -83.399
      assert(np.abs ( psiPrev - val_160731 ) < 0.001 ), "FAIL!" 

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
    if(arg=="-test1"): 
      #test1(np.float(sys.argv[1]),np.float(sys.argv[2]),np.float(sys.argv[3]))
      test1(np.float(sys.argv[1]),np.float(sys.argv[2]))
      quit()


  daIter()      #np.float(sys.argv[1]),np.float(sys.argv[2]),np.float(sys.argv[3]))

