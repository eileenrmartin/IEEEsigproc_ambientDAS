myPathToCode = /home/ermartin/StanfordDASArray/IEEEsigproc_ambientDAS/code
#myPathToCode = /CorrelationCode

pythonCPPInterfaces:
	python setup.py build_ext --inplace

testMods:	
	make pythonCPPInterfaces
	python testCorrs.py


startSpecCh = 71
endSpecCh = 80
specPlot:
	python calc_spec.py ${startSpecCh} ${endSpecCh} ${myPathToCode}/src/params/params 1 2
	python plot_spec.py ${startSpecCh} ${endSpecCh} ${myPathToCode}/src/params/params 1 2 

oneBitXCorr%:
	python calc_oneBitXCorr.py ${myPathToCode}/src/params/params$* baseline

filteredOneBitXCorr%:
	python calc_oneBitXCorr.py ${myPathToCode}/src/params/params$* filtered

plotOneBitXCorr%:
	python plot_oneBitXCorr.py ${myPathToCode}/src/params/params$* baseline


plotOneBitXCorrs%:
	python plot_oneBitXCorrs_multi_file_lists.py ${myPathToCode}/src/params/params$*plot rasterTestPlot
	
trainingPrep%: 
	python training_prep.py params/trainingparams$*

training: 
	python training.py params/trainingparams1

clean:
	rm *.so






oneBitXCorrWeek1:
	make oneBitXCorr1
	make oneBitXCorr2
	make oneBitXCorr3
	make oneBitXCorr4
	make oneBitXCorr5
	make oneBitXCorr6
	make oneBitXCorr7

oneBitXCorrWeek2:
	make oneBitXCorr8
	make oneBitXCorr9
	make oneBitXCorr10
	make oneBitXCorr11
	make oneBitXCorr12
	make oneBitXCorr13
	make oneBitXCorr14

oneBitXCorrWeek3:
	make oneBitXCorr15
	make oneBitXCorr16
	make oneBitXCorr17
	make oneBitXCorr18
	make oneBitXCorr19
	make oneBitXCorr20
	make oneBitXCorr21

oneBitXCorrWeek4:
	#make oneBitXCorr22
	#make oneBitXCorr23
	#make oneBitXCorr24
	#make oneBitXCorr25
	#make oneBitXCorr26
	#make oneBitXCorr27
	make oneBitXCorr28
	make oneBitXCorr29
	make oneBitXCorr30

