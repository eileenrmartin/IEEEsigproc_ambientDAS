myPathToCode = /home/ermartin/StanfordDASArray/IEEEsigproc_ambientDAS/code
#myPathToCode = /CorrelationCode

pythonCPPInterfaces:
	python setup.py build_ext --inplace

oneBitXCorr%:
	python calc_oneBitXCorr.py ${myPathToCode}/src/params/params%

plotOneBitXCorr%:
	python plot_oneBitXCorr.py ${myPathToCode}/src/params/params%


plotOneBitXCorrs%:
	python plot_oneBitXCorrs_multi_file_lists.py ${myPathToCode}/src/params/params%plot rasterTestPlot



