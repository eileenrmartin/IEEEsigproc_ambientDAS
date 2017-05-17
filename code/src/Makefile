myPathToCode = /home/ermartin/StanfordDASArray/IEEEsigproc_ambientDAS/code
#myPathToCode = /CorrelationCode

pythonCPPInterfaces:
	python setup.py build_ext --inplace

testMods:	
	make pythonCPPInterfaces
	python testCorrs.py

oneBitXCorr%:
	python calc_oneBitXCorr.py ${myPathToCode}/src/params/params$* baseline

filteredOneBitXCorr%:
	python calc_oneBitXCorr.py ${myPathToCode}/src/params/params$* filtered

plotOneBitXCorr%:
	python plot_oneBitXCorr.py ${myPathToCode}/src/params/params$*


plotOneBitXCorrs%:
	python plot_oneBitXCorrs_multi_file_lists.py ${myPathToCode}/src/params/params$*plot rasterTestPlot
	
trainingPrep%: 
	python training_prep.py params/trainingparams$*

training: 
	python training.py params/trainingparams1

clean:
	rm *.so

