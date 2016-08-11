#!/usr/bin/env python
import sys
##################################
#
# Revisions
#       10.08.10 inception
#
##################################

#
# ROUTINE  
#
import MSAtools as msa
import numpy as np
#msa.idxNa = 2
idxOxy = 0
idxCl = 1
idxNa = 2
idxCa = 3
nOxy = 8.
ref_conc_M = np.array([1e-200, (100.0e-3), 100.0e-3, 0.0])  # [M] order is  Oxy, Cl, Na, Ca (bath) 
zs = np.array([-0.5, -1.0, 1.0, 2.0])
Ns = np.array([8., 0, 0, 1.,]) # Filter
V_i = 0.375 #3.4 angtstrom radius from VMD
#V_i = 0.1131 #3.0 angstrom radius
sigmas = np.array([0.278, 0.362, 0.204, 0.200])
epsilonFilter = 63.5


# validation at low Ca  
def validation1(): 
  cacl2 = 1e-10
  conc_M = np.copy(ref_conc_M)
  conc_M[idxCl] = ref_conc_M[idxCl]+2 * cacl2
  conc_M[idxCa] = cacl2
   
  chemPotential,donnanPotential,muES,muHS,rhoFilter = msa.SolveMSAEquations(epsilonFilter,conc_M,zs,Ns,V_i,sigmas)
  correctmu = np.array([4.7113072,6.90965741,1.22106921,-4.1851689])
  correctpsi = -173.100988931
  correctrhos = np.array([2.13333333e+01,4.88143119e-08,1.06652933e+01,6.72892874e-04])
  eps = 1e-3
  if np.abs(all(chemPotential) - all(correctmu))<eps:
     print "SUCCESS MUS!"
  else:
     print "FAIL MUS"
  if np.abs(donnanPotential-correctpsi)<eps:
     print "SUCCESS PSI!"
  else:
     print "FAIL PSI!"
     print donnanPotential
  if np.abs(all(rhoFilter) - all(correctrhos))<eps:
     print "SUCCESS RHOS!"
  else:
     print "FAIL RHOS!"


validation1()
#validation2()

#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation 1" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
#if __name__ == "__main__":
#  import sys
#  msg = helpmsg()
#  remap = "none"

#  if len(sys.argv) < 2:
#      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
#  for i,arg in enumerate(sys.argv):
#    # calls 'doit' with the next argument following the argument '-validation'
#    if(arg=="-validation"):
#      arg1=np.int(sys.argv[i+1])
#      doit(arg1)
  





#  raise RuntimeError("Arguments not understood")




