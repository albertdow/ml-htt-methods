import ROOT
from optparse import OptionParser
import os


parser = OptionParser()
parser.add_option("--filelist", dest="filelist",
                  help="" % vars())
parser.add_option("--dir", dest="dir",
                  help="" % vars())

(options, args) = parser.parse_args()

dir = options.dir

count=1

maxperjobs=300000

with open(options.filelist) as fp: 
    for line in fp:
        l = line.strip()
        #if not ('VBFHToTauTauUncorrelatedDecay_Filtered' in l or 'TauF' in l) or 'Embed' in l: continue # temp while we dont want to run on all files
        if 'Higgs' in l or 'JJH0' in l: continue # temp while we dont want to run on all files
        print l
        if '.root' not in l: continue
        if os.path.isfile('%(dir)s/%(l)s'%vars()):
          f = ROOT.TFile('%(dir)s/%(l)s'%vars())
          t = f.Get('ntuple')
          entries=t.GetEntries()
          nsplit=0
          for i in range(0,entries,maxperjobs):
            outF = open("x%(count)i" % vars(), "w")
            out_line = '%(l)s %(nsplit)i' % vars() 
            nsplit+=1
            outF.write(out_line)
            outF.close() 
            count+=1
       

