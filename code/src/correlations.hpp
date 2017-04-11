#include "boost/multiarray.hpp"
#include <cmath>

double correlationCoeff(boost::multi_array<float, 1> &longTermFct, boost::multi_array<float, 1> &shortTermFct){
  /// Calculates the zero time lag correlation 
  /// but does so with blocked sums (so no overflow)

  double corrCoeff = 0.0;
  float partialSum = 0.0;
  int len = ****;

  // break arrays into blocks
  for(int i=0; i<len; ++i){
    partialSum += longTermFct[i] * shortTermFct[i];
    if(std::abs(partialSum) > 0.25*FLT_MAX){  // *****check what max floating point value is named******
      corrCoeff += double(partialSum);
      partialSum = 0;
    }
  }
  // add any stragler terms at the end
  corrCoeff += double(partialSum);

  return corrCoeff;
}


boost::multi_array<int, 1> xCorr1Bit(boost::multi_array<char, 1> &rec, boost::multi_array<char, 1> &src){
	boost::multi_array<int, 1> xCorr; 
	// ****declare size
	
	

}

